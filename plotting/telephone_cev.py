"""Telephone CEV panels with across-seed min-max bands (paper Fig. panels a,b).

Each empirical CEV curve is the mean over 5 seeds — the canonical run
(seed 77) plus the 4 seed reruns — with a min-max band across seeds. The
probe batch is sampled with the same fixed key for every run
(`feedback_runs.SEED`), so the theory references (factored / Kronecker) are
identical across seeds and drawn once; only the model activations vary.

Ported verbatim from the frozen `figure_telephone_cev_seeds.py`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from .lib.feedback_runs import (
    acts_at_last_resid_post, cev_of_rows, kron_per_token, load_cfg_proc,
    load_model, sample_batch, stack_beliefs,
)
from .lib.runs import resolve_run

N_SHOW = 20
CEV_STEPS = [0, 500, 5000, 50000]

# The paper's two telephone families (canonical seed-77 run + 4 seed reruns).
PANELS = [
    ["telephone_fractal2_sym_canonical_50k"]
    + [f"telephone_fractal2_sym_canonical_s{s}_50k" for s in (801, 802, 803, 804)],
    ["telephone_mess3_canonical_50k"]
    + [f"telephone_mess3_canonical_s{s}_50k" for s in (401, 402, 403, 404)],
]


def cev_run(run_dir: Path, want_theory: bool):
    """Per-step empirical CEV (+ theory refs from the first run only)."""
    run_dir = Path(run_dir)
    cfg, process = load_cfg_proc(run_dir)
    inputs, beliefs = sample_batch(cfg, process)
    out = {"cev_per_step": {}}
    for step in CEV_STEPS:
        ckpt = run_dir / f"checkpoints/{step}/model.pt"
        if not ckpt.exists():
            raise FileNotFoundError(ckpt)
        m = load_model(cfg, ckpt)
        a = acts_at_last_resid_post(m, inputs)[:, 1:]
        out["cev_per_step"][step] = cev_of_rows(a.reshape(-1, a.shape[-1]))
    if want_theory:
        b_list = stack_beliefs(beliefs)
        factored = np.concatenate([b[:, 1:] for b in b_list], axis=-1)
        out["cev_factored"] = cev_of_rows(factored.reshape(-1, factored.shape[-1]))
        u_b = [b[:, 1:] for b in b_list[1:]]
        kron = kron_per_token(u_b)
        out["cev_kron"] = cev_of_rows(kron.reshape(-1, kron.shape[-1]))
        out["dim_factored"] = int(sum(b.shape[-1] for b in b_list))
        out["dim_kron"] = int(np.prod([b.shape[-1] for b in b_list[1:]]))
    print(f"  {run_dir.name}: done")
    return out


def collect_panel(run_names):
    run_dirs = [resolve_run(r) for r in run_names]
    runs = [cev_run(rd, want_theory=(i == 0)) for i, rd in enumerate(run_dirs)]
    panel = {k: runs[0][k] for k in ("cev_factored", "cev_kron", "dim_factored", "dim_kron")}
    panel["n_seeds"] = len(runs)
    panel["mean"], panel["min"], panel["max"] = {}, {}, {}
    for step in CEV_STEPS:
        stack = np.stack([r["cev_per_step"][step][:N_SHOW] for r in runs])
        panel["mean"][step] = stack.mean(0)
        panel["min"][step] = stack.min(0)
        panel["max"][step] = stack.max(0)
    return panel


def _short_step(step: int) -> str:
    """500 -> '0.5k', 50000 -> '50k' -- keeps the legend narrow."""
    if step < 1000:
        return str(step)
    return f"{step / 1000:g}k"


def draw_cev_panel(ax, p, title, rs=None, cmap=None, fs=8, compact=False,
                   xtick_step=2):
    """Draw one telephone CEV panel on `ax`. Shared with the combined figure.

    compact: shorten the legend (short step labels, no seed-count title) for
    use in multi-panel figures rendered at a single column/text width.
    xtick_step: label every nth rank. Every other rank is ten labels across a
    cell-sized panel, which reads as a grey band rather than as numbers.
    """
    rs = np.arange(1, N_SHOW + 1) if rs is None else rs
    cmap = plt.cm.viridis if cmap is None else cmap
    n_steps = len(CEV_STEPS)
    for i, step in enumerate(CEV_STEPS):
        color = cmap(0.12 + 0.78 * i / max(1, n_steps - 1))
        ax.fill_between(rs, p["min"][step], p["max"][step], color=color,
                        alpha=0.25, lw=0)
        lab = f"step {_short_step(step)}" if compact else f"step {step}"
        ax.plot(rs, p["mean"][step], "-o", lw=1.0, ms=2.2, color=color,
                label=lab)
    # compact (paper) panels name the references without their simplex
    # dimensions: the numbers are total dimensions, not CEV quantities
    lab_f = "factored" if compact else f"factored (dim={p['dim_factored']})"
    lab_k = "Kronecker" if compact else f"Kronecker (dim={p['dim_kron']})"
    # same encoding as the rank panel (plotting.effective_rank_vs_n):
    # factored = dashed, Kronecker = dotted, black, no point symbols
    r = np.arange(1, len(p["cev_factored"]) + 1)[:N_SHOW]
    ax.plot(r, p["cev_factored"][:N_SHOW], "--", lw=1.5, color="black",
            label=lab_f)
    ax.plot(r, p["cev_kron"][:N_SHOW], ":", lw=1.7, color="black",
            label=lab_k)
    ax.axhline(0.8, ls=":", lw=0.4, color="gray")
    ax.axhline(0.95, ls=":", lw=0.4, color="gray")
    ax.set_ylim(0, 1.02); ax.set_xlim(0.5, N_SHOW + 0.5)
    ax.set_xticks(np.arange(xtick_step, N_SHOW + 1, xtick_step))
    ax.set_xlabel("rank $r$", fontsize=fs)
    ax.set_title(title, fontsize=fs + 1)
    ax.tick_params(labelsize=fs - 2)
    leg = ax.legend(loc="lower right", fontsize=fs - 2, framealpha=0.92,
                    handlelength=1.2, borderpad=0.3, labelspacing=0.25,
                    title=None if compact else
                    f"mean; band = min–max, {p['n_seeds']} seeds")
    if not compact:
        leg.get_title().set_fontsize(fs - 2.5)
