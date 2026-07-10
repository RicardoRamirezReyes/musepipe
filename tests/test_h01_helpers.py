import numpy as np

from musepipe.extraction.product import FORMAT_VERSION, SpectrumProduct
from musepipe.stages.stage_h01_detect import HALPHA_REST_A, expected_line_center_A, gaussian_flux_template


def h01_wave():
    return np.arange(6525.0, 6601.0, 1.0, dtype=np.float64)


def h01_product(method="psffit", *, signal_flux=0.0, rv_sys_kms=0.0, lsf_fwhm_A=2.5):
    wave = h01_wave()
    center = expected_line_center_A(HALPHA_REST_A, rv_sys_kms)
    flux = np.zeros(wave.size, dtype=np.float64)
    if signal_flux:
        flux += float(signal_flux) * gaussian_flux_template(wave, center, lsf_fwhm_A)
    fits_method = "optimal" if method.startswith("optimal") else method
    return SpectrumProduct(
        wave_A=wave,
        flux=flux,
        flux_err=np.full(wave.size, 0.5, dtype=np.float64),
        flux_err_emp=np.full(wave.size, 0.5, dtype=np.float64),
        apcorr=np.ones(wave.size, dtype=np.float64),
        npix_eff=np.full(wave.size, 9.0, dtype=np.float64),
        flags=np.zeros(wave.size, dtype=np.int32),
        header={
            "FORMATV": FORMAT_VERSION,
            "METHOD": fits_method,
            "RUNID": "synthetic_h01",
            "SRCPOS_Y": 12.0,
            "SRCPOS_X": 16.0,
            "APERTURE": method,
            "WFRAME": "barycentric",
            "INCUBE": "cube.fits",
            "INCUBESH": "samecube",
            "NORMRAD": 25.0,
            "ERRMODE": "total",
            "APCMODE": "psf_model_norm_radius",
        },
        extra_columns={
            "cont_runmed": np.zeros(wave.size, dtype=np.float64),
            "flux_err_total": np.full(wave.size, 0.5, dtype=np.float64),
        },
    )


def h01_controls(n_controls=100, noise=0.0, seed=1):
    wave = h01_wave()
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, float(noise), size=(int(n_controls), wave.size))
