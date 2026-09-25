"""MLflow file-backend parsing helpers for the loss-curve figures.

The training runs log through MLflow's file backend; these readers parse its
append-only metric files and params trees directly, with no MLflow client
dependency.
"""
from __future__ import annotations

from pathlib import Path


def mlflow_load_metric_series(metric_path: Path) -> list[tuple[int, float, int]]:
    """Parse an MLflow file-backend metric file (append-only
    `<timestamp_ms> <value> <step>` lines). Returns list of triples."""
    metric_path = Path(metric_path)
    if not metric_path.exists():
        raise FileNotFoundError(metric_path)
    out: list[tuple[int, float, int]] = []
    for line in metric_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) != 3:
            continue
        try:
            out.append((int(parts[0]), float(parts[1]), int(parts[2])))
        except (ValueError, TypeError):
            continue
    return out


def mlflow_metric_at_final_step(metric_path: Path) -> float | None:
    """Return the value at the highest step. If two rows share that step
    (e.g. a "resumed run" where the final-step log line fired twice), keep
    the *latest by timestamp*. Returns None if the file is empty / missing."""
    try:
        rows = mlflow_load_metric_series(metric_path)
    except FileNotFoundError:
        return None
    if not rows:
        return None
    max_step = max(s for _, _, s in rows)
    final_rows = [(ts, v) for ts, v, s in rows if s == max_step]
    final_rows.sort(key=lambda r: r[0])
    return final_rows[-1][1]


def mlflow_load_params(params_dir: Path) -> dict[str, str]:
    """Walk an MLflow file-backend params/ directory and return a flat dict
    of `dotted.path` -> string value. Nested params dirs become dotted keys."""
    params_dir = Path(params_dir)
    if not params_dir.exists():
        raise FileNotFoundError(params_dir)
    out: dict[str, str] = {}
    for p in sorted(params_dir.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(params_dir)
        key = ".".join(rel.parts)
        try:
            out[key] = p.read_text().rstrip("\n")
        except (OSError, UnicodeDecodeError):
            continue
    return out
