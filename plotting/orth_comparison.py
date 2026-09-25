"""Appendix figure: unprojected vs encoder-orthogonalized steering writes.

  (a) gap closed per setup and patched factor, for the unprojected write
      (filled) and the encoder-orthogonalized write of eq:steering_write
      (open), joined by a line. Means over seeds (ratio of means, as in the
      main-text panels); small dots show individual seeds.
  (b) the candidate explainer: between-factor decoder subspace overlap
      (how much of the edited factor's encoder span lies in the partners')
      against the gap difference none - encoder. The overlap is substantial
      everywhere (0.13-0.58) yet barely correlates with the sensitivity to
      the projection -- the sns loops are the outliers in BOTH directions --
      so the overlap does not explain which writes are fragile.

Reads steering_orth_variants.json.

Library module: make_figure(out_dir) writes figure_orth_comparison.{png,pdf}.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .lib.runs import REPO  # noqa: E402
JSON_PATH = REPO / "analysis" / "steering_orth_variants.json"
FIG_W = 5.5
FS, FS_LAB, FS_TICK, FS_LEG = 7.5, 8.0, 6.0, 5.5

GROUPS = [
    ("loop_mess3", "loop mess3$\\,\\leftrightarrow\\,$mess3", "#1f6f8b"),
    ("loop_fractal2_mess3", "loop fractal2$\\,\\to\\,$mess3", "#2e8b57"),
    ("loop_mess3_fractal2", "loop mess3$\\,\\to\\,$fractal2", "#2e8b57"),
    ("loop_sns_mess3", "loop sns$\\,\\to\\,$mess3", "#c0392b"),
    ("loop_mess3_sns", "loop mess3$\\,\\to\\,$sns", "#c0392b"),
    ("telephone_mess3", "tel. $3\\times$mess3", "#b8860b"),
    ("telephone_fractal2", "tel. $5\\times$fractal2", "#7d5ba6"),
]
FACTOR_NAME = {  # per group: factor index -> short label
    "loop_mess3": {0: "$T$", 1: "$U$"},
    "loop_fractal2_mess3": {0: "$T$ (f2)", 1: "$U$ (m3)"},
    "loop_mess3_fractal2": {0: "$T$ (m3)", 1: "$U$ (f2)"},
    "loop_sns_mess3": {0: "$T$ (sns)", 1: "$U$ (m3)"},
    "loop_mess3_sns": {0: "$T$ (m3)", 1: "$U$ (sns)"},
}


def rows_for(res, key):
    rs = [r for r in res if r["group"] == key]
    out = []
    for fi in sorted({f["factor"] for r in rs for f in r["factors"]}):
        fs = [f for r in rs for f in r["factors"] if f["factor"] == fi]
        gap = {}
        seeds = {}
        for k in ("none", "encoder"):
            gap[k] = 1 - (np.mean([f["schemes"][k]["measured"] for f in fs])
                          / np.mean([f["null"] for f in fs]))
            seeds[k] = [1 - f["schemes"][k]["measured"] / f["null"] for f in fs]
        out.append(dict(factor=fi, gap=gap, seeds=seeds,
                        overlap=float(np.mean([f["decoder_overlap"] for f in fs]))))
    return out


def make_figure(out_dir: Path) -> None:
    res = json.loads(JSON_PATH.read_text())

    fig, (ax, axb) = plt.subplots(
        1, 2, figsize=(FIG_W, 3.1), gridspec_kw={"width_ratios": [1.25, 1]})

    # ---- (a) dumbbells ------------------------------------------------------
    ylabels, y = [], 0
    for key, label, colour in GROUPS:
        for rec in rows_for(res, key):
            g0, g1 = rec["gap"]["none"], rec["gap"]["encoder"]
            ax.plot([100 * g0, 100 * g1], [y, y], "-", color=colour, lw=1.0,
                    alpha=0.7, zorder=2)
            ax.plot([100 * g0], [y], "o", color=colour, ms=4, zorder=3)
            ax.plot([100 * g1], [y], "o", mfc="white", mec=colour, ms=4,
                    mew=1.1, zorder=3)
            for k, mk in (("none", "o"),):
                pass
            # per-seed dots, tiny, both schemes
            for k, dy in (("none", 0.18), ("encoder", -0.18)):
                v = 100 * np.asarray(rec["seeds"][k])
                ax.plot(v, np.full(v.shape, y + dy), ".", ms=1.6, color=colour,
                        alpha=0.55, zorder=2)
            fname = FACTOR_NAME.get(key, {}).get(
                rec["factor"], f"$U_{rec['factor']}$")
            ylabels.append(f"{label},  patch {fname}")
            y += 1
        y += 0.6                                   # gap between groups
    ax.axvline(100, color="0.75", lw=0.7, zorder=1)
    ax.axvline(0, color="0.75", lw=0.7, ls=":", zorder=1)
    yt = []
    yy = 0
    for key, _, _ in GROUPS:
        n = len(rows_for(res, key))
        yt.extend(list(np.arange(yy, yy + n)))
        yy += n + 0.6
    ax.set_yticks(yt)
    ax.set_yticklabels(ylabels, fontsize=FS_TICK - 0.5)
    ax.invert_yaxis()
    ax.set_xlabel("gap closed  [%]", fontsize=FS_LAB)
    ax.tick_params(labelsize=FS_TICK)
    ax.grid(alpha=0.25, axis="x")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    handles = [Line2D([], [], marker="o", ls="none", color="0.3",
                      label="unprojected write"),
               Line2D([], [], marker="o", ls="none", mfc="white", mec="0.3",
                      label="encoder-orthogonalized")]
    ax.legend(handles=handles, fontsize=FS_LEG, loc="lower left",
              framealpha=0.93, borderpad=0.3, handletextpad=0.4)
    ax.text(-0.60, 1.03, "(a)", transform=ax.transAxes, fontsize=FS_LAB,
            fontweight="bold", ha="left", va="bottom")

    # ---- (b) overlap vs sensitivity -----------------------------------------
    for key, label, colour in GROUPS:
        for rec in rows_for(res, key):
            drop = 100 * (rec["gap"]["none"] - rec["gap"]["encoder"])
            axb.scatter([rec["overlap"]], [drop], s=22, color=colour,
                        zorder=3)
    axb.axhline(0, color="0.75", lw=0.7, zorder=1)
    axb.set_xlabel("decoder subspace overlap\n(edited factor vs. partners)",
                   fontsize=FS_LAB)
    axb.set_ylabel("gap difference, unprojected $-$ orth.  [pp]",
                   fontsize=FS_LAB)
    axb.tick_params(labelsize=FS_TICK)
    axb.grid(alpha=0.25)
    for side in ("top", "right"):
        axb.spines[side].set_visible(False)
    # family legend (colour key), compact
    fam = [("mess3 loop", "#1f6f8b"), ("fractal2 loops", "#2e8b57"),
           ("sns loops", "#c0392b"), ("tel. $3\\times$m3", "#b8860b"),
           ("tel. $5\\times$f2", "#7d5ba6")]
    axb.legend(handles=[Line2D([], [], marker="o", ls="none", color=c, label=l)
                        for l, c in fam],
               fontsize=FS_LEG, loc="lower left", framealpha=0.93,
               borderpad=0.3, handletextpad=0.4)
    axb.text(-0.28, 1.03, "(b)", transform=axb.transAxes, fontsize=FS_LAB,
             fontweight="bold", ha="left", va="bottom")

    fig.subplots_adjust(left=0.305, right=0.985, top=0.94, bottom=0.19,
                        wspace=0.42)
    out = Path(out_dir) / "figure_orth_comparison.png"
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"wrote {out} (+.pdf)")
