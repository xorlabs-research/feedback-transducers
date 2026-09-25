"""Patched-output KL against its original / null / unrelated controls.

For each patching setup, four KLs on the JOINT next-token distribution (nats):

  unedited || original     the model's own fidelity when untouched -- the floor
  patched  || mixed        the measurement: did the splice produce the target?
  unedited || mixed        the null -- what a write with no effect would give
  unedited || unrelated    the scale -- distance to a wholly unrelated sequence

One point per (run, edited factor); the bar is the mean. Reads
analysis/steering_kl_controls_paper.json (written by
analysis/steering_kl_controls_paper.py).

Ported verbatim from the frozen `figure_steering_kl_controls_paper.py`.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D

from .lib.runs import REPO

JSON_PATH = REPO / "analysis" / "steering_kl_controls_paper.json"

FS_LAB = 8.0     # axis labels
FS_TICK = 6.0    # tick labels
FS_LEG = 5.5     # legend entries

# key -> (legend label, colour, marker face)
QUANTITIES = [
    ("baseline", "unedited $\\|$ original  (model's own floor)", "0.45", True),
    ("measured", "patched $\\|$ mixed  (the edit)", "#1f6f8b", True),
    ("control_null", "unedited $\\|$ mixed  (null: write had no effect)",
     "#b8860b", False),
    ("control_rand", "unedited $\\|$ unrelated  (scale)", "#c0392b", False),
]
# short per-bar labels, in QUANTITIES order, for narrow single-group panels
# where a legend does not fit (bar_labels=True)
BAR_SHORT = ["floor", "edit", "null", "unrel."]
SETUP_ORDER = ["loop_mess3", "telephone_mess3", "telephone_fractal2"]
SETUP_LABELS = {"loop_mess3": "loop\nmess3$\\,\\leftrightarrow\\,$mess3",
                "telephone_mess3": "telephone\n$3\\times$ mess3",
                "telephone_fractal2": "telephone\n$5\\times$ fractal2",
                "loop_sns_mess3": "loop\n$T{=}$sns, $U{=}$mess3",
                "loop_mess3_sns": "loop\n$T{=}$mess3, $U{=}$sns"}


def load(json_path: Path = JSON_PATH) -> dict:
    """setup key -> (per-intervention rows, number of runs).

    Every setup in the json, not just SETUP_ORDER: the sns figure draws the two
    sns loops from the same file.
    """
    res = json.loads(Path(json_path).read_text())
    per_setup = {}
    for key in dict.fromkeys(r["setup"] for r in res):
        rows = [e for r in res if r["setup"] == key for e in r["edits"]]
        n_runs = len({r["run"] for r in res if r["setup"] == key})
        per_setup[key] = (rows, n_runs)
    return per_setup


def draw_controls_panel(ax, per_setup, fs_lab=FS_LAB, fs_tick=FS_TICK,
                        fs_leg=FS_LEG, legend=True, xlabels=True, order=None,
                        ymax=6.0, gap_y=1.6, bar_labels=False,
                        group_labels=None):
    """Draw the four-KL bar panel on `ax`. Shared with the combined figures, so
    the standalone and combined versions cannot drift apart.

    order: which setups to show, left to right (default: the three that steer).
    bar_labels: label each BAR on the x axis instead of each group, and drop the
        legend -- for a narrow single-group panel that has no room for one.
    group_labels: x tick labels for the groups, overriding SETUP_LABELS. Lets a
        caller group by something other than setup (e.g. which belief is
        patched) by passing synthesized per-group keys.
    """
    order = SETUP_ORDER if order is None else order
    offs = np.linspace(-0.3, 0.3, len(QUANTITIES))
    width = 0.16

    for gi, key in enumerate(order):
        rows, _ = per_setup[key]
        for (qkey, _, colour, filled), dx in zip(QUANTITIES, offs):
            v = np.array([e[qkey] for e in rows])
            x = gi + dx
            ax.bar(x, np.mean(v), width=width, color=colour,
                   alpha=0.85 if filled else 0.35,
                   edgecolor=colour, linewidth=0.8, zorder=2)
            # every intervention, so the spread across seeds and factors is
            # visible rather than hidden behind the mean
            jit = np.linspace(-width * 0.3, width * 0.3, len(v))
            ax.plot(x + jit, v, ".", ms=1.8, color="0.15", alpha=0.8, zorder=3,
                    ls="none")

        # how much of the null-to-target distance the edit actually closed.
        # gap_y=None suppresses it, for callers that place it themselves.
        if gap_y is not None:
            mm = np.mean([e["measured"] for e in rows])
            nn = np.mean([e["control_null"] for e in rows])
            # fixed height, so the annotations align across groups
            ax.text(gi, gap_y, f"gap closed {100 * (1 - mm / nn):.1f}%",
                    ha="center", va="bottom", fontsize=fs_tick,
                    fontweight="bold", color="0.2")

    ax.set_yscale("log")
    ax.set_ylim(2e-5, ymax)
    ax.set_xlim(-0.55, len(order) - 0.45)
    if bar_labels:
        ax.set_xticks([offs[j] + gi for gi in range(len(order))
                       for j in range(len(QUANTITIES))])
        ax.set_xticklabels(BAR_SHORT * len(order), fontsize=fs_tick,
                           rotation=60, ha="right")
        legend = False
    else:
        ax.set_xticks(range(len(order)))
        names = (group_labels if group_labels is not None
                 else [SETUP_LABELS[k] for k in order])
        ax.set_xticklabels(names if xlabels else [""] * len(order),
                           fontsize=fs_lab)
    ax.set_ylabel("KL to reference  [nats]", fontsize=fs_lab)
    ax.tick_params(labelsize=fs_tick)
    ax.tick_params(axis="x", length=0)
    ax.grid(alpha=0.3, axis="y", which="major", zorder=0)
    ax.set_axisbelow(True)

    # all swatches solid: the four colours already distinguish the entries, and
    # hollow squares read as a separate category rather than as the lighter bar
    # fill they were meant to echo
    handles = [Line2D([], [], marker="s", ls="none", ms=4, mfc=c, mec=c,
                      label=lab)
               for _, lab, c, _ in QUANTITIES]
    handles.append(Line2D([], [], marker=".", ls="none", ms=3, color="0.15",
                          label="one intervention (seed $\\times$ factor)"))
    if legend:
        # above the axes: inside, the legend would cover the (very small)
        # baseline bars of the telephone setups
        ax.legend(handles=handles, fontsize=fs_leg, loc="lower center", ncol=3,
                  bbox_to_anchor=(0.5, 1.01), framealpha=0.0, handlelength=1.0,
                  labelspacing=0.3, columnspacing=1.0, borderpad=0.2,
                  handletextpad=0.5)
    return handles
