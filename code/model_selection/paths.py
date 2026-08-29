from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunOutputPaths:
    """Filesystem locations produced by one model-selection run."""

    root: Path
    model_comparison: Path
    hyperparameter_search: Path
    hyperparameter_raw: Path
    feature_selection: Path
    explainability: Path
    explainability_raw: Path
    diagnostics: Path
    figures: Path


def create_run_output_paths(
    base_output_dir: Path,
    run_tag: str,
) -> RunOutputPaths:
    """Create and return the semantic output directories for one run."""
    root = base_output_dir / run_tag
    paths = RunOutputPaths(
        root=root,
        model_comparison=root / "model_comparison",
        hyperparameter_search=root / "hyperparameter_search",
        hyperparameter_raw=root / "hyperparameter_search" / "raw",
        feature_selection=root / "feature_selection",
        explainability=root / "explainability",
        explainability_raw=root / "explainability" / "raw",
        diagnostics=root / "diagnostics",
        figures=root / "figures",
    )

    for directory in (
        paths.model_comparison,
        paths.hyperparameter_search,
        paths.hyperparameter_raw,
        paths.feature_selection,
        paths.explainability,
        paths.explainability_raw,
        paths.diagnostics,
        paths.figures,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    return paths
