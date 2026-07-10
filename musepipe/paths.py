"""Central path helpers for run directories and stage products."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunPaths:
    """Resolved filesystem paths for one pipeline run."""

    project_root: Path
    run_id: str
    runs_dir: Path
    run_dir: Path
    config_dir: Path
    stage_dir: Path
    plot_dir: Path
    table_dir: Path
    log_dir: Path

    @classmethod
    def from_project_root(
        cls,
        run_id: str,
        project_root: str | Path | None = None,
        runs_dir_name: str = "runs",
    ) -> "RunPaths":
        root = Path("." if project_root is None else project_root).resolve()
        runs_dir = root / runs_dir_name
        run_dir = runs_dir / str(run_id)
        return cls(
            project_root=root,
            run_id=str(run_id),
            runs_dir=runs_dir,
            run_dir=run_dir,
            config_dir=run_dir / "config",
            stage_dir=run_dir / "stages",
            plot_dir=run_dir / "plots",
            table_dir=run_dir / "tables",
            log_dir=run_dir / "logs",
        )

    @property
    def config_json(self) -> Path:
        return self.config_dir / "config.json"

    def plot_stage_dir(self, stage_name: str) -> Path:
        return self.plot_dir / str(stage_name)

    def ensure_base_dirs(self, include_logs: bool = True) -> None:
        dirs = [self.config_dir, self.stage_dir, self.plot_dir, self.table_dir]
        if include_logs:
            dirs.append(self.log_dir)
        for directory in dirs:
            directory.mkdir(parents=True, exist_ok=True)


__all__ = ["RunPaths"]
