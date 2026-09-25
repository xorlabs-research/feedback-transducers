"""Shared run-loading primitives for the plotting tools.

Run dirs live under `<repo>/experiments/<figure>/runs/<run>/` and contain
`hydra/config.yaml` + `checkpoints/<step>/model.pt`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
# Data root: the directory holding experiments/<figure>/runs/<run>/ (with each
# run's checkpoints/ and mlruns/ dropped in). Defaults to the repo itself;
# point FACTORED_REPS_DATA elsewhere to analyze artifacts living outside the repo.
import os
DATA_ROOT = Path(os.environ.get("FACTORED_REPS_DATA", REPO))
# run name -> run dir relative to the data root, indexed from the frozen
# configs committed in this repo (the data root mirrors this layout).
RUN_DIRS: dict[str, Path] = {
    p.parents[1].name: p.parents[1].relative_to(REPO)
    for p in sorted((REPO / "experiments").glob("*/runs/*/hydra/config.yaml"))
}
# Make fwh_core + the training package importable for hydra instantiation.
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def run_path(name: str) -> Path:
    """Map a run name to its dir under the data root."""
    if name not in RUN_DIRS:
        raise FileNotFoundError(f"unknown run {name!r}: no experiments/*/runs/{name}/hydra/config.yaml in the repo")
    return DATA_ROOT / RUN_DIRS[name]


def resolve_run(run: str) -> Path:
    """Map a run name (or path) to its run dir."""
    p = Path(run)
    if p.is_dir() and (p / "hydra").is_dir():
        return p.resolve()
    cand = run_path(run)
    if not (cand / "hydra" / "config.yaml").exists():
        raise FileNotFoundError(f"no hydra/config.yaml under {cand}")
    return cand


def _latest_step(ckpt_dir: Path) -> int:
    steps = sorted(int(p.name) for p in ckpt_dir.iterdir()
                   if p.is_dir() and p.name.isdigit() and (p / "model.pt").exists())
    if not steps:
        raise FileNotFoundError(f"no checkpoints/<step>/model.pt under {ckpt_dir}")
    return steps[-1]


def available_steps(run: str | Path) -> list[int]:
    ckpt_dir = resolve_run(str(run)) / "checkpoints"
    return sorted(int(p.name) for p in ckpt_dir.iterdir()
                  if p.is_dir() and p.name.isdigit() and (p / "model.pt").exists())


def load_run(run: str | Path, step: int | None = None):
    """Instantiate process + model from the saved hydra config, load a checkpoint.

    step=None -> step 50000 if present, else the latest available checkpoint.
    Returns (cfg, process, model) with everything on CPU.
    """
    from omegaconf import OmegaConf
    import hydra

    run_dir = resolve_run(str(run))
    if not OmegaConf.has_resolver("hydra"):
        OmegaConf.register_new_resolver("hydra", lambda key: f"<hydra:{key}>")
    cfg = OmegaConf.load(run_dir / "hydra" / "config.yaml")
    cfg.device = "cpu"
    process = hydra.utils.instantiate(cfg.generative_process.instance)
    base_vocab = int(process.vocab_size)
    cfg.generative_process.vocab_size = base_vocab + 1
    cfg.generative_process.base_vocab_size = base_vocab
    cfg.generative_process.bos_token = base_vocab
    if OmegaConf.is_missing(cfg.generative_process, "eos_token"):
        cfg.generative_process.eos_token = None
    cfg.predictive_model.instance.cfg.d_vocab = base_vocab + 1
    cfg.predictive_model.instance.cfg.device = "cpu"
    model = hydra.utils.instantiate(cfg.predictive_model.instance)

    ckpt_dir = run_dir / "checkpoints"
    if step is None:
        step = 50000 if (ckpt_dir / "50000" / "model.pt").exists() else _latest_step(ckpt_dir)
    ckpt = ckpt_dir / str(step) / "model.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"no checkpoint at {ckpt}")
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    elif isinstance(state, dict):
        model.load_state_dict(state)
    model.eval()
    return cfg, process, model


def sample_batch(cfg, process, batch_size: int = 2048, seed: int = 12345):
    """Sample one fixed batch from the (deterministic) process.

    Returns dict with inputs [B,T] (torch.long) and belief_states (tuple of
    per-factor [B,T,d]) + the full generate_data_batch_with_full_history outs.
    """
    import jax
    from fwh_core.generative_processes.torch_generator import (
        generate_data_batch_with_full_history,
    )
    from experiments.training.utils import expand_state_by_batch_size

    bos = cfg.generative_process.bos_token
    eos = cfg.generative_process.eos_token
    n_ctx = cfg.predictive_model.instance.cfg.n_ctx
    seq_len = n_ctx - int(bos is not None) - int(eos is not None)
    init = expand_state_by_batch_size(process.initial_state, batch_size)
    outs = generate_data_batch_with_full_history(
        init, process, batch_size, seq_len, jax.random.PRNGKey(seed),
        bos_token=bos, eos_token=eos,
    )
    return outs
