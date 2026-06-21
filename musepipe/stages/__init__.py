"""Stage entry points for the refactored pipeline."""

from .stage04b_local_surface import run_stage04b, stage04b_config_from_run
from .stage06_local_surface_injection import run_stage06_local, stage06_local_config_from_run
from .stage06_local_surface_sweep import run_stage06_local_sweep, stage06_local_sweep_config_from_run
from .stage06_pca_c2_summary import run_stage06_pca_c2_summary
from .stage07_accretion_lines import run_stage07, stage07_config_from_run
from .stage07b_halpha_robustness import run_stage07b, stage07b_config_from_run

__all__ = [
    "run_stage04b",
    "run_stage06_local",
    "run_stage06_local_sweep",
    "run_stage06_pca_c2_summary",
    "run_stage07",
    "run_stage07b",
    "stage04b_config_from_run",
    "stage06_local_config_from_run",
    "stage06_local_sweep_config_from_run",
    "stage07_config_from_run",
    "stage07b_config_from_run",
]
