import tempfile
import unittest
from pathlib import Path

import numpy as np

from musepipe.extraction.product import FORMAT_VERSION, SpectrumProduct


def valid_header():
    return {
        "FORMATV": FORMAT_VERSION,
        "METHOD": "aperture",
        "RUNID": "synthetic",
        "SRCPOS_Y": 12.5,
        "SRCPOS_X": 14.5,
        "APERTURE": "box3",
        "WFRAME": "topocentric",
        "INCUBE": "cube.fits",
        "INCUBESH": "abc123",
        "NORMRAD": 18.0,
    }


def valid_product(**header_overrides):
    header = valid_header()
    header.update(header_overrides)
    wave = np.linspace(6500.0, 6504.0, 5)
    return SpectrumProduct(
        wave_A=wave,
        flux=np.arange(5, dtype=np.float64),
        flux_err=np.full(5, 0.2),
        flux_err_emp=np.full(5, 0.3),
        apcorr=np.full(5, 1.1),
        npix_eff=np.full(5, 9.0),
        flags=np.arange(5, dtype=np.int32),
        header=header,
    )


class SpectrumProductTests(unittest.TestCase):
    def test_roundtrip_preserves_columns_and_validates(self):
        product = valid_product()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "spec.fits"
            product.write(path)
            loaded = SpectrumProduct.read(path)

        np.testing.assert_allclose(loaded.wave_A, product.wave_A)
        np.testing.assert_allclose(loaded.flux, product.flux)
        np.testing.assert_allclose(loaded.apcorr, product.apcorr)
        np.testing.assert_array_equal(loaded.flags, product.flags)
        self.assertEqual(loaded.header["FORMATV"], FORMAT_VERSION)
        self.assertEqual(loaded.header["METHOD"], "aperture")

    def test_validator_rejects_missing_required_header(self):
        product = valid_product()
        header = dict(product.header)
        header.pop("INCUBESH")
        malformed = SpectrumProduct(
            wave_A=product.wave_A,
            flux=product.flux,
            flux_err=product.flux_err,
            flux_err_emp=product.flux_err_emp,
            apcorr=product.apcorr,
            npix_eff=product.npix_eff,
            flags=product.flags,
            header=header,
        )
        with self.assertRaisesRegex(ValueError, "missing required"):
            malformed.validate()

    def test_validator_rejects_unknown_format_version(self):
        product = valid_product(FORMATV=999)
        with self.assertRaisesRegex(ValueError, "Unsupported FORMATV"):
            product.validate()

    def test_roundtrip_preserves_extra_columns(self):
        product = valid_product()
        product = SpectrumProduct(
            wave_A=product.wave_A,
            flux=product.flux,
            flux_err=product.flux_err,
            flux_err_emp=product.flux_err_emp,
            apcorr=product.apcorr,
            npix_eff=product.npix_eff,
            flags=product.flags,
            header=product.header,
            extra_columns={"flux_err_total": np.arange(5, dtype=np.float64) + 1.0},
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "spec_extra.fits"
            product.write(path)
            loaded = SpectrumProduct.read(path)

        self.assertIn("flux_err_total", loaded.extra_columns)
        np.testing.assert_allclose(loaded.extra_columns["flux_err_total"], product.extra_columns["flux_err_total"])


if __name__ == "__main__":
    unittest.main()
