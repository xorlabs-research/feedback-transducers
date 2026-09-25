"""Complexification through coarse-graining (paper Fig. figure_coarse_graining_cev).

CEV by rank, one color per observed interface: DASHED = exact belief
representation (theory), SOLID = trained model's final-layer residual
stream; lines are means over the five replicates, bands their min--max.

Reads coarse_graining_trained.json (written by
`analysis/coarse_graining_trained.py`).

Ported verbatim from the frozen `figure_coarse_graining.py`.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D

MODES = [("full", "full $(x,y)$", "#1f6f8b"),
         ("x", "$x$ only (agent $U$ hidden)", "#b03a2e"),
         ("y", "$y$ only (agent $T$ hidden)", "#d68910")]
N_SHOW = 40


def _load(json_path: Path):
    res = json.loads(Path(json_path).read_text())
    reps = sorted({r["replicate"] for r in res})
    by = {(r["replicate"], r["mode"]): r for r in res}
    return reps, by


def panel_cev(ax, reps, by, panel_label=None, short_labels=False):
    """Theory (dashed) vs trained model (solid), mean + min-max band.

    short_labels: drop "agent" from the hidden-factor entries. The legend sits
    over the curves in a cell-sized panel, so its width is worth the terseness.
    """
    for mode, label, c in MODES:
        rows = [by[(rep, mode)] for rep in reps]
        for key, ls, lw in (("cev_belief", "--", 1.2), ("cev_model", "-", 1.6)):
            # A curve is only as long as its object's dimension: the
            # full-observation factored concatenation is 20-dim (5 + 15), so
            # its spectrum has 20 values, while the coarse joint belief (75-dim)
            # and the residual stream (120-dim) have more. CEV is trivially 1
            # beyond an object's dimension, so pad to the axis limit rather
            # than letting the line stop mid-plot.
            M = np.stack([np.pad(np.array(r[key])[:N_SHOW],
                                 (0, max(0, N_SHOW - len(r[key]))),
                                 constant_values=1.0) for r in rows])
            xs = np.arange(1, M.shape[1] + 1)
            ax.plot(xs, M.mean(0), ls, lw=lw, color=c, alpha=0.95)
            ax.fill_between(xs, M.min(0), M.max(0), color=c, alpha=0.15, lw=0)
    ax.axhline(0.95, ls=":", lw=0.7, color="0.5")
    ax.text(N_SHOW, 0.955, "95%", fontsize=6, color="0.4", ha="right", va="bottom")
    ax.set_xlabel("rank $r$", fontsize=8)
    ax.set_ylabel("cumulative explained variance", fontsize=8)
    ax.set_xlim(1, N_SHOW); ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3); ax.tick_params(labelsize=7)
    if panel_label:
        ax.text(0.02, 0.97, panel_label, transform=ax.transAxes, fontsize=10,
                fontweight="bold", va="top")
    handles = [Line2D([], [], color=c, lw=1.6,
                      label=label.replace("agent ", "") if short_labels else label)
               for _, label, c in MODES]
    handles += [Line2D([], [], color="0.35", lw=1.6, label="trained model"),
                Line2D([], [], color="0.35", lw=1.2, ls="--", label="exact belief")]
    ax.legend(handles=handles, fontsize=6.5, loc="lower right", framealpha=0.93,
              handlelength=1.6, labelspacing=0.25)
