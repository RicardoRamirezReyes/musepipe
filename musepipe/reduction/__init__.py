"""Raw-reduction helpers for ESO/MUSE pipeline entry points."""

__all__ = [
    "ReductionError",
    "build_inventory",
    "check_esorex_environment",
    "ensure_a1_run_tree",
    "write_inventory_csv",
]


def __getattr__(name):
    if name in __all__:
        from . import esorex_driver

        return getattr(esorex_driver, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
