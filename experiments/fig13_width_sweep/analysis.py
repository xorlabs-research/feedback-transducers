"""Effective rank vs. residual-stream width for the telephone mess3 N=3 process.

Appendix control for the reviewer objection behind fig:telephone_cev(c):
*does the measured effective rank track the generative PROCESS or the
ARCHITECTURE?* One process (telephone mess3, N=3), three model widths
(d_model 60 / 120 / 240), five seeds each. If the rank tracks the process it
sits flat near the factored bound $\\sum_i (d_i-1)$ regardless of width; if it
tracks the architecture it grows with d_model (the dotted $\\propto d_{model}$
guide).

Metrics, all on the BOS-excluded final-layer `resid_post` at step 50000
(same activation and threshold convention as `plotting/effective_rank_vs_n.py`,
whose `_cev`/`_nc` this mirrors so the d120 numbers reproduce that figure):

  nc@p  smallest #PCs whose cumulative explained variance reaches p (p=.80,.95)
  PR    participation ratio (Σλ)² / Σλ² of the PCA eigenvalues — a
        threshold-free effective dimension, included because nc@p can be
        pushed around by a long tail of near-zero directions that a wider
        residual stream supplies for free.

Usage (from the repo root, with the runs' checkpoints in place; see REPRODUCING.md):

    python experiments/fig13_width_sweep/analysis.py

Results are cached per run in results.json, so re-running after fetching more
seeds only computes the new ones (--refresh recomputes everything). Missing
runs are skipped with a warning, so the figure can be drawn from partial data
while the chain is still going.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from plotting.lib.runs import available_steps, load_run, sample_batch  # noqa: E402

FINAL_STEP = 50000

# d_model -> run names. The d120 column is the canonical telephone mess3
# family (seed 77 + s401..s404); d60/d240 use seeds 2001..2005 / 2101..2103
# (see experiments/SEED_MANIFEST.md).
RUNS: dict[int, list[str]] = {
    60: [f"telephone_mess3_d60_canonical_s{s}_50k" for s in range(2001, 2006)],
    120: ["telephone_mess3_canonical_50k"]
         + [f"telephone_mess3_canonical_s{s}_50k" for s in (401, 402, 403, 404)],
    240: [f"telephone_mess3_d240_canonical_s{s}_50k" for s in range(2101, 2104)],
}

_FACTORED_C = "#b03a2e"
_JOINT_C = "#1f6f8b"
_ARCH_C = "#6c6c6c"


# ---- metrics ---------------------------------------------------------------

def _eigs(acts: np.ndarray) -> np.ndarray:
    """PCA eigenvalues (descending) of the BOS-excluded activations."""
    flat = acts[:, 1:, :].reshape(-1, acts.shape[-1])  # skip BOS
    Xc = flat - flat.mean(0, keepdims=True)
    s = np.linalg.svd(Xc, compute_uv=False)
    return s ** 2


def _nc(cev: np.ndarray, p: float) -> int:
    """Smallest #PCs reaching cumulative variance p (1-indexed)."""
    return int(np.searchsorted(cev, p) + 1)


def _participation_ratio(var: np.ndarray) -> float:
    """(Σλ)² / Σλ². 1 for a single dominant direction, d for isotropic."""
    return float(var.sum() ** 2 / max(float((var ** 2).sum()), 1e-30))


def measure(run: str, batch: int, seed: int, thresholds: list[float]) -> dict:
    step_used = FINAL_STEP if FINAL_STEP in available_steps(run) else max(available_steps(run))
    cfg, process, model = load_run(run, step=step_used)
    n_layers = int(cfg.predictive_model.instance.cfg.n_layers)
    d_model = int(cfg.predictive_model.instance.cfg.d_model)
    hook = f"blocks.{n_layers - 1}.hook_resid_post"

    outs = sample_batch(cfg, process, batch, seed)
    inputs = torch.as_tensor(np.asarray(outs["inputs"]), dtype=torch.long)
    with torch.no_grad():
        _, cache = model.run_with_cache(inputs, return_type="logits")
    var = _eigs(cache[hook].cpu().numpy())
    cev = np.cumsum(var / max(var.sum(), 1e-30))

    # Closed-form theory dims from the per-factor belief dims:
    # factored = Σ(d_i-1), joint = Πd_i - 1.
    beliefs = outs["belief_states"]
    if not isinstance(beliefs, tuple):
        beliefs = (beliefs,)
    dims = [int(b.shape[-1]) for b in beliefs]
    return {
        "run": run, "d_model": d_model, "n_layers": n_layers,
        "step": step_used, "preliminary": step_used < FINAL_STEP,
        "nc": {str(p): _nc(cev, p) for p in thresholds},
        "pr": _participation_ratio(var),
        "dim_factored": int(sum(d - 1 for d in dims if d > 1)),
        "dim_joint": int(np.prod(dims)) - 1,
    }


# ---- collection ------------------------------------------------------------

def collect(batch: int, seed: int, thresholds: list[float], cache_path: Path,
            refresh: bool) -> dict[str, dict]:
    cache: dict[str, dict] = {}
    if cache_path.exists() and not refresh:
        cache = json.loads(cache_path.read_text()).get("runs", {})
    changed = False
    for d_model, runs in sorted(RUNS.items()):
        for run in runs:
            key = f"{run}|b{batch}|s{seed}"
            if key in cache:
                print(f"  cached  d={d_model:3d} {run}")
                continue
            try:
                r = measure(run, batch, seed, thresholds)
            except FileNotFoundError:
                print(f"  SKIP    d={d_model:3d} {run}: run not available")
                continue
            changed = True
            cache[key] = r
            print(f"  measured d={d_model:3d} {run}: nc={r['nc']} PR={r['pr']:.2f}"
                  f"{'  [PRELIM step %d]' % r['step'] if r['preliminary'] else ''}",
                  flush=True)
            cache_path.write_text(json.dumps(
                {"batch": batch, "seed": seed, "thresholds": thresholds,
                 "runs": cache}, indent=2))
    if changed:
        cache_path.write_text(json.dumps(
            {"batch": batch, "seed": seed, "thresholds": thresholds,
             "runs": cache}, indent=2))
    return cache


def group(cache: dict[str, dict]) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = {}
    by_run = {r["run"]: r for r in cache.values()}
    for d_model, runs in sorted(RUNS.items()):
        got = [by_run[r] for r in runs if r in by_run]
        if got:
            out[d_model] = got
    return out


# ---- figure ----------------------------------------------------------------

def _band(ax, xs, series, color, label, marker="o"):
    """mean line + min-max band over seeds."""
    mean = np.array([np.mean(v) for v in series])
    lo = np.array([np.min(v) for v in series])
    hi = np.array([np.max(v) for v in series])
    ax.plot(xs, mean, "-", color=color, lw=1.8, marker=marker, ms=5, label=label)
    if (hi > lo).any():
        ax.fill_between(xs, lo, hi, color=color, alpha=0.2, lw=0)
    return mean


def draw_panel(ax, grouped: dict[int, list[dict]], thresholds: list[float],
               title: bool = True, fs: float = 10):
    widths = np.array(sorted(grouped))
    res = [grouped[d] for d in widths]
    factored = res[0][0]["dim_factored"]
    joint = res[0][0]["dim_joint"]

    # theory references: horizontal, because the PROCESS does not change across
    # the sweep — that flatness is the whole point of the panel.
    ax.axhline(joint, ls="-.", color=_JOINT_C, lw=2.0,
               label=f"joint-Kronecker bound $\\prod_i d_i - 1$ = {joint}")
    ax.axhline(factored, ls="--", color=_FACTORED_C, lw=2.2,
               label=f"factored bound $\\sum_i (d_i-1)$ = {factored}")
    cmap = plt.get_cmap("viridis")
    for j, p in enumerate(thresholds):
        c = cmap(0.15 + 0.6 * j / max(1, len(thresholds) - 1))
        _band(ax, widths, [[r["nc"][str(p)] for r in g] for g in res], c,
              f"empirical effective rank ($n_c$@{int(p*100)}%)")
    # participation ratio is still computed and cached in results.json /
    # printed in the table; it is deliberately not drawn.

    prelim = [any(r.get("preliminary") for r in g) for g in res]
    for d, g, pl in zip(widths, res, prelim):
        if pl:
            y = max(max(r["nc"][str(max(thresholds))] for r in g), 1)
            ax.annotate("preliminary\n(mid-training ckpt)", (d, y),
                        xytext=(-8, 14), textcoords="offset points",
                        ha="right", fontsize=7.5, color="0.25")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(widths)
    ax.set_xticklabels([str(int(d)) for d in widths])
    ax.minorticks_off()
    ax.set_xlabel("residual-stream width $d_{\\mathrm{model}}$", fontsize=fs)
    ax.set_ylabel("dimension  (log scale)", fontsize=fs)
    n_seeds = ", ".join(f"d{int(d)}: {len(g)}" for d, g in zip(widths, res))
    if title:
        ax.set_title("Residual-stream belief-state dimension tracks the process, not the width\n"
                     f"(telephone, 3 mess3 sub-agents; seeds — {n_seeds}; bands = min–max)",
                     fontsize=10)
    ax.grid(alpha=0.3, which="both")
    # headroom above the Kronecker line so the 6-row legend has somewhere to sit
    # that isn't on top of the empirical curves at the bottom of the panel
    ax.set_ylim(top=joint * 4.0)
    ax.legend(fontsize=fs - 2.5, loc="upper left", framealpha=0.93)


def make_figure(grouped, out: Path, thresholds, title: bool = True):
    fig, ax = plt.subplots(figsize=(4.6, 3.5))
    draw_panel(ax, grouped, thresholds, title=title)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out} (+.pdf)")


def print_table(grouped, thresholds):
    print("\n  d_model | seeds |" + "".join(f"  nc@{int(p*100)}%      |" for p in thresholds)
          + "  PR            | factored | joint")
    for d, g in sorted(grouped.items()):
        cells = ""
        for p in thresholds:
            v = np.array([r["nc"][str(p)] for r in g], dtype=float)
            cells += f" {v.mean():5.1f} [{v.min():.0f},{v.max():.0f}] |"
        pr = np.array([r["pr"] for r in g])
        print(f"  {d:7d} | {len(g):5d} |{cells} {pr.mean():5.2f} "
              f"[{pr.min():.2f},{pr.max():.2f}] | {g[0]['dim_factored']:8d} | {g[0]['dim_joint']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", type=int, default=4096,
                    help="sequences sampled from the process (default matches "
                         "plotting/effective_rank_vs_n.py)")
    ap.add_argument("--seed", type=int, default=12345,
                    help="sampling seed; identical batch for every run")
    ap.add_argument("--thresholds", default="0.80,0.95")
    ap.add_argument("--cache", default=str(HERE / "results.json"))
    ap.add_argument("--refresh", action="store_true", help="ignore the cache")
    ap.add_argument("--out", default=str(HERE / "effective_rank_vs_width.png"))
    ap.add_argument("--no-title", action="store_true",
                    help="omit the axes title (paper version; caption in LaTeX)")
    args = ap.parse_args()

    thresholds = [float(t) for t in args.thresholds.split(",")]
    cache = collect(args.batch, args.seed, thresholds, Path(args.cache), args.refresh)
    grouped = group(cache)
    if len(grouped) < 2:
        print(f"  only {len(grouped)} width(s) available — need >=2 for the sweep figure")
        return
    print_table(grouped, thresholds)
    make_figure(grouped, Path(args.out), thresholds, title=not args.no_title)


if __name__ == "__main__":
    main()
