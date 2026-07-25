"""Stage entry points for the refactored pipeline."""

from .stage01_align import run_stage01, stage01_config_from_run
from .stage01c_localize import run_stage01c, stage01c_config_from_run
from .stage02_xcorr import run_stage02, stage02_config_from_run
from .stage04b_local_surface import run_stage04b, stage04b_config_from_run
from .stage06_local_surface_injection import run_stage06_local, stage06_local_config_from_run
from .stage06_local_surface_sweep import run_stage06_local_sweep, stage06_local_sweep_config_from_run
from .stage06_pca_c2_summary import run_stage06_pca_c2_summary
from .stage07_accretion_lines import run_stage07, stage07_config_from_run
from .stage07b_halpha_robustness import run_stage07b, stage07b_config_from_run
from .stage08c_look_elsewhere import run_stage08c, stage08c_config_from_run
from .stage_h01_detect import run_stage_h01, stage_h01_config_from_run
from .stage_h02_artifacts import run_stage_h02, stage_h02_config_from_run
from .stage_h03_limits import run_stage_h03, stage_h03_config_from_run
from .stage_h04_injection import run_stage_h04, stage_h04_config_from_run
from .stage_e01_psf import run_stage_e01, stage_e01_config_from_run
from .stage_x01_aperture import run_stage_x01, stage_x01_config_from_run
from .stage_x02_optimal import run_stage_x02, stage_x02_config_from_run
from .stage_x03_psffit import run_stage_x03, stage_x03_config_from_run
from .stage_x10_compare import run_stage_x10, stage_x10_config_from_run
from .stage_x11_calibrate import (
    definitive_spectra_figure,
    halo_remaining_figure,
    halo_removal_figure,
    run_stage_x11,
    stage_x11_config_from_run,
    unsubtracted_aperture_reference,
)

__all__ = [
    "definitive_spectra_figure",
    "halo_remaining_figure",
    "halo_removal_figure",
    "unsubtracted_aperture_reference",
    "run_stage01",
    "run_stage01c",
    "run_stage02",
    "run_stage04b",
    "run_stage06_local",
    "run_stage06_local_sweep",
    "run_stage06_pca_c2_summary",
    "run_stage07",
    "run_stage07b",
    "run_stage08c",
    "run_stage_h01",
    "run_stage_h02",
    "run_stage_h03",
    "run_stage_h04",
    "run_stage_e01",
    "run_stage_x01",
    "run_stage_x02",
    "run_stage_x03",
    "run_stage_x10",
    "run_stage_x11",
    "stage01_config_from_run",
    "stage01c_config_from_run",
    "stage02_config_from_run",
    "stage04b_config_from_run",
    "stage06_local_config_from_run",
    "stage06_local_sweep_config_from_run",
    "stage07_config_from_run",
    "stage07b_config_from_run",
    "stage08c_config_from_run",
    "stage_h01_config_from_run",
    "stage_h02_config_from_run",
    "stage_h03_config_from_run",
    "stage_h04_config_from_run",
    "stage_e01_config_from_run",
    "stage_x01_config_from_run",
    "stage_x02_config_from_run",
    "stage_x03_config_from_run",
    "stage_x10_config_from_run",
    "stage_x11_config_from_run",
]
