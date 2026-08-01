"""Frozen standard spectrum product format for extraction methods."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits


FORMAT_VERSION = 1
SPECTRUM_HDU = "SPECTRUM"
REQUIRED_COLUMNS = (
    "wave_A",
    "flux",
    "flux_err",
    "flux_err_emp",
    "apcorr",
    "npix_eff",
    "flags",
)
REQUIRED_HEADERS = (
    "FORMATV",
    "METHOD",
    "RUNID",
    "SRCPOS_Y",
    "SRCPOS_X",
    "APERTURE",
    "WFRAME",
    "INCUBE",
    "INCUBESH",
    "NORMRAD",
)


@dataclass(frozen=True)
class SpectrumProduct:
    wave_A: np.ndarray
    flux: np.ndarray
    flux_err: np.ndarray
    flux_err_emp: np.ndarray
    apcorr: np.ndarray
    npix_eff: np.ndarray
    flags: np.ndarray
    header: dict
    covariance: np.ndarray | None = None
    extra_columns: dict[str, np.ndarray] | None = None

    def as_table_hdu(self):
        columns = [
            fits.Column(name="wave_A", format="D", array=np.asarray(self.wave_A, dtype=np.float64)),
            fits.Column(name="flux", format="D", array=np.asarray(self.flux, dtype=np.float64)),
            fits.Column(name="flux_err", format="D", array=np.asarray(self.flux_err, dtype=np.float64)),
            fits.Column(name="flux_err_emp", format="D", array=np.asarray(self.flux_err_emp, dtype=np.float64)),
            fits.Column(name="apcorr", format="D", array=np.asarray(self.apcorr, dtype=np.float64)),
            fits.Column(name="npix_eff", format="D", array=np.asarray(self.npix_eff, dtype=np.float64)),
            fits.Column(name="flags", format="J", array=np.asarray(self.flags, dtype=np.int32)),
        ]
        for name, values in (self.extra_columns or {}).items():
            if name in REQUIRED_COLUMNS:
                raise ValueError(f"Extra column duplicates required column: {name}")
            arr = np.asarray(values)
            fmt = "J" if np.issubdtype(arr.dtype, np.integer) else "D"
            columns.append(fits.Column(name=str(name), format=fmt, array=arr))
        hdu = fits.BinTableHDU.from_columns(columns, name=SPECTRUM_HDU)
        for key, value in self._fits_header_items().items():
            hdu.header[key] = value
        return hdu

    def _fits_header_items(self):
        hdr = dict(self.header)
        hdr["FORMATV"] = int(hdr.get("FORMATV", FORMAT_VERSION))
        for key in ("METHOD", "RUNID", "APERTURE", "WFRAME", "INCUBE", "INCUBESH"):
            if key in hdr and hdr[key] is not None:
                hdr[key] = str(hdr[key])
        for key in ("SRCPOS_Y", "SRCPOS_X", "NORMRAD"):
            if key in hdr and hdr[key] is not None:
                hdr[key] = float(hdr[key])
        return hdr

    def validate(self) -> None:
        arrays = [
            np.asarray(self.wave_A),
            np.asarray(self.flux),
            np.asarray(self.flux_err),
            np.asarray(self.flux_err_emp),
            np.asarray(self.apcorr),
            np.asarray(self.npix_eff),
            np.asarray(self.flags),
        ]
        if any(arr.ndim != 1 for arr in arrays):
            raise ValueError("SpectrumProduct columns must be 1D arrays.")
        n = arrays[0].size
        if any(arr.size != n for arr in arrays):
            raise ValueError("SpectrumProduct columns must have matching lengths.")
        if n == 0:
            raise ValueError("SpectrumProduct must contain at least one row.")
        if int(self.header.get("FORMATV", FORMAT_VERSION)) != FORMAT_VERSION:
            raise ValueError(f"Unsupported FORMATV={self.header.get('FORMATV')!r}.")
        missing = [key for key in REQUIRED_HEADERS if key not in self.header]
        if missing:
            raise ValueError(f"SpectrumProduct header missing required keys: {missing}")
        # "sgf"/"lpm": spectral-diversity halo-subtraction methods (specs C5/C6,
        # docs/2026-07-15_plan_integracion_halosub.md); additive to the contract.
        if str(self.header["METHOD"]) not in {"aperture", "optimal", "psffit", "hrsdi", "sgf", "lpm"}:
            raise ValueError(f"Invalid METHOD={self.header['METHOD']!r}.")
        if np.any(~np.isfinite(np.asarray(self.wave_A, dtype=float))):
            raise ValueError("wave_A contains non-finite values.")
        if np.any(np.asarray(self.apcorr, dtype=float) <= 0):
            raise ValueError("apcorr must be positive.")
        if self.covariance is not None and self.covariance.shape != (n, n):
            raise ValueError("COVARIANCE shape must be (N,N).")
        for name, values in (self.extra_columns or {}).items():
            arr = np.asarray(values)
            if arr.ndim != 1 or arr.size != n:
                raise ValueError(f"Extra column {name!r} must be 1D and match spectrum length.")

    def write(self, path, overwrite=True):
        self.validate()
        hdus = [fits.PrimaryHDU(), self.as_table_hdu()]
        if self.covariance is not None:
            hdus.append(fits.ImageHDU(np.asarray(self.covariance, dtype=np.float64), name="COVARIANCE"))
        fits.HDUList(hdus).writeto(path, overwrite=overwrite)
        return Path(path)

    @classmethod
    def read(cls, path):
        with fits.open(path, memmap=False) as hdul:
            if SPECTRUM_HDU not in hdul:
                raise ValueError(f"Missing {SPECTRUM_HDU} HDU.")
            table = hdul[SPECTRUM_HDU].data
            names = set(table.names)
            missing = [name for name in REQUIRED_COLUMNS if name not in names]
            if missing:
                raise ValueError(f"SPECTRUM table missing columns: {missing}")
            header = _science_header_items(hdul[SPECTRUM_HDU].header)
            covariance = hdul["COVARIANCE"].data.astype(np.float64) if "COVARIANCE" in hdul else None
            extra_columns = {
                name: np.asarray(table[name]).copy()
                for name in table.names
                if name not in REQUIRED_COLUMNS
            }
            product = cls(
                wave_A=np.asarray(table["wave_A"], dtype=np.float64),
                flux=np.asarray(table["flux"], dtype=np.float64),
                flux_err=np.asarray(table["flux_err"], dtype=np.float64),
                flux_err_emp=np.asarray(table["flux_err_emp"], dtype=np.float64),
                apcorr=np.asarray(table["apcorr"], dtype=np.float64),
                npix_eff=np.asarray(table["npix_eff"], dtype=np.float64),
                flags=np.asarray(table["flags"], dtype=np.int32),
                header=header,
                covariance=covariance,
                extra_columns=extra_columns,
            )
        product.validate()
        return product


def read_spectrum_product(path) -> SpectrumProduct:
    return SpectrumProduct.read(path)


def _science_header_items(header):
    structural = {
        "XTENSION",
        "BITPIX",
        "NAXIS",
        "NAXIS1",
        "NAXIS2",
        "PCOUNT",
        "GCOUNT",
        "TFIELDS",
        "EXTNAME",
        "CHECKSUM",
        "DATASUM",
    }
    out = {}
    for key in header:
        if key in ("", "COMMENT", "HISTORY"):
            continue
        if key in structural:
            continue
        if key.startswith(("TTYPE", "TFORM", "TUNIT", "TDIM", "TNULL", "TSCAL", "TZERO")):
            continue
        out[key.replace("HIERARCH ", "")] = header[key]
    return out


__all__ = [
    "FORMAT_VERSION",
    "REQUIRED_COLUMNS",
    "REQUIRED_HEADERS",
    "SPECTRUM_HDU",
    "SpectrumProduct",
    "read_spectrum_product",
]
