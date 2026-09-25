"""Convergence to the Bayes-optimal floor (the paper's convergence figure).

Eval loss over training (absolute, not the gap), with each family's
Bayes-optimal floor drawn as a dashed hline in the family color, for (a) the
four main-text processes (the two mess3-fractal2 feedback loops and the two
telephones) and (b) the telephone-mess3 N-sweep family. Every curve is the
across-seed MEAN over the 5 seeds (canonical run + 4 seeds; see
experiments/SEED_MANIFEST.md), with a min–max band.

Data source: each run's MLflow store,
<data_root>/experiments/<figure>/runs/<run>/mlruns/.

The floor is a Monte-Carlo estimate (32768 samples, ~1e-3-nat precision);
per-seed floors differ by ~1e-3 nats, so the hline drawn is the family mean.

The PAPER uses the symlog-x variant (xscale="symlog").
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (FixedLocator, FuncFormatter, LogLocator,
                               NullFormatter)

from .lib.eval_lib import mlflow_load_metric_series
from .lib.runs import run_path


def _symlog_fmt(v, _pos):
    if v == 0:
        return "0"
    return r"$10^{%d}$" % int(round(np.log10(v)))


def _seed_family(base: str, seeds: list[int]) -> list[str]:
    return [base] + [f"{base}_s{s}" for s in seeds]


PANEL_A = [  # (family run names, label, color)
    # the mess3-fractal2 loops of the belief-geometry figures; legend letters
    # as in the paper's figure
    (_seed_family("feedback_mess3_fractal2_canonical", [1301, 1302, 1303, 1304]),
     "feedback A$'$ ($T$=mess3, $U$=fractal2)", "#4C72B0"),
    (_seed_family("feedback_fractal2_mess3_canonical", [1001, 1002, 1003, 1004]),
     "feedback B$'$ ($T$=fractal2, $U$=mess3)", "#DD8452"),
    (_seed_family("telephone_fractal2_sym_canonical", [801, 802, 803, 804]),
     "telephone, 5$\\times$fractal2", "#55A868"),
    (_seed_family("telephone_mess3_canonical", [401, 402, 403, 404]),
     "telephone, 3$\\times$mess3", "#8172B3"),
]
PANEL_B = [
    (_seed_family("telephone_mess3_n2_canonical", [501, 502, 503, 504]), "$N=2$"),
    (_seed_family("telephone_mess3_canonical", [401, 402, 403, 404]), "$N=3$"),
    (_seed_family("telephone_mess3_n4_canonical", [601, 602, 603, 604]), "$N=4$"),
    (_seed_family("telephone_mess3_n5_canonical", [701, 702, 703, 704]), "$N=5$"),
]


def _find_run(name: str) -> Path:
    """The MLflow run dir of run `name` in its own store."""
    store = run_path(f"{name}_50k") / "mlruns"
    runs = [d for d in store.glob("*/*") if (d / "metrics/eval/loss/step").exists()]
    if len(runs) != 1:
        raise FileNotFoundError(f"expected one mlflow run with an eval series under {store}, found {len(runs)}")
    return runs[0]


def loss_series(name: str):
    run = _find_run(name)
    triples = mlflow_load_metric_series(run / "metrics/eval/loss/step")
    triples = sorted(triples, key=lambda t: t[2])
    steps = np.array([t[2] for t in triples])
    loss = np.array([t[1] for t in triples])
    floor = float((run / "params/optimal_loss/average").read_text())
    return steps, loss, floor


def family_loss(names: list[str]):
    """Across-seed mean / min / max of the LOSS, on the common step grid, plus
    the per-seed floors. Seeds whose mlflow data is not available are
    skipped with a warning."""
    series = []
    for n in names:
        try:
            series.append(loss_series(n))
        except FileNotFoundError as e:
            print(f"  WARNING: skipping {n}: {e}")
    if not series:
        raise FileNotFoundError(f"no runs available for family {names[0]}")
    steps0 = series[0][0]
    gaps = []
    for st, gp, _ in series:
        if len(st) == len(steps0) and np.array_equal(st, steps0):
            gaps.append(gp)
        else:  # union grids differ (shouldn't happen) — interpolate
            gaps.append(np.interp(steps0, st, gp))
    G = np.stack(gaps)
    return steps0, G.mean(0), G.min(0), G.max(0), [s[2] for s in series]


def make(xscale: str, out: Path, verbose: bool = True):
    """Two panels — (a) main-text runs, (b) N-sweep — absolute eval loss on a
    log y-axis, each family's Bayes floor as a dashed hline in its color.
    xscale 'linear' or 'symlog' (compact linear 0..10^3 segment, log beyond).
    The paper figure is the symlog variant."""
    out = Path(out)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.0))
    for names, label, color in PANEL_A:
        steps, mean, lo, hi, floors = family_loss(names)
        ax1.plot(steps, mean, lw=1.5, color=color, label=label)
        ax1.fill_between(steps, lo, hi, color=color, alpha=0.22, lw=0)
        ax1.axhline(float(np.mean(floors)), ls="--", lw=1.0, color=color, alpha=0.85)
        if verbose:
            print(f"  A {names[0]} (n={len(names)}): floor={np.mean(floors):.4f}, "
                  f"final loss={mean[-1]:.4f} [{lo[-1]:.4f},{hi[-1]:.4f}]")
    cmap = plt.get_cmap("viridis")
    for i, (names, label) in enumerate(PANEL_B):
        steps, mean, lo, hi, floors = family_loss(names)
        c = cmap(0.12 + 0.7 * i / (len(PANEL_B) - 1))
        ax2.plot(steps, mean, lw=1.5, color=c, label=label)
        ax2.fill_between(steps, lo, hi, color=c, alpha=0.22, lw=0)
        ax2.axhline(float(np.mean(floors)), ls="--", lw=1.0, color=c, alpha=0.85)
        if verbose:
            print(f"  B {names[0]} (n={len(names)}): floor={np.mean(floors):.4f}, "
                  f"final loss={mean[-1]:.4f} [{lo[-1]:.4f},{hi[-1]:.4f}]")
    for ax in (ax1, ax2):
        ax.plot([], [], "--", lw=1.0, color="0.35", label="Bayes-optimal floor")
    for ax, ttl in ((ax1, "main-text runs"), (ax2, "telephone mess3, $N$-sweep")):
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(subs=(1.0, 1.5, 2.0, 3.0, 5.0, 7.0)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
        if xscale == "symlog":
            # symlog x: compact linear 0..10^3 segment, log axis from 10^3 on.
            ax.set_xscale("symlog", linthresh=1000, linscale=0.5)
            ax.set_xlim(0, 52000)
            ax.xaxis.set_major_locator(FixedLocator([0, 1e3, 1e4]))
            ax.xaxis.set_major_formatter(FuncFormatter(_symlog_fmt))
        else:
            ax.set_xlim(0, 51000)
            ax.set_xticks([0, 10000, 20000, 30000, 40000, 50000])
            ax.set_xticklabels(["0", "10k", "20k", "30k", "40k", "50k"])
        ax.set_xlabel("optimizer step")
        ax.set_title(ttl, fontsize=9)
        ax.grid(alpha=0.3, which="both")
        ax.tick_params(which="both", labelsize=6.5)
        ax.legend(fontsize=6.5, framealpha=0.92,
                  title="mean; band = min–max, 5 seeds", title_fontsize=5.5)
    ax1.set_ylabel("eval loss (nats)")
    ax1.text(0.02, 0.97, "(a)", transform=ax1.transAxes, fontsize=10,
             fontweight="bold", va="top")
    ax2.text(0.02, 0.97, "(b)", transform=ax2.transAxes, fontsize=10,
             fontweight="bold", va="top")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200); fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"wrote {out} (+.pdf)")
