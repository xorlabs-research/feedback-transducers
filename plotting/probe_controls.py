"""Combined probe-controls appendix figure.

Top row: probe selectivity controls (true / time-shifted / shuffled targets)
over training, one panel per feedback setup.
Bottom row: layer-wise emergence, one panel per (setup, factor), one curve
per residual-stream depth.

All panels share the same convention: held-out probe error 1 - R^2 on a log
y-axis, optimizer step on a symlog x-axis (step 0 visible). Every curve is the
mean over five seeds per setup (the canonical run plus four reruns); the
shaded band is the min-max envelope across those seeds.

Data: the JSONs produced by
  experiments/fig11_probe_controls/probe_selectivity_controls/probe_selectivity_controls.py
  experiments/fig11_probe_controls/probe_layerwise_emergence/compute_layerwise_r2.py

Library module: make_figure(out_dir) writes figure_probe_controls.{pdf,png}.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .lib.runs import REPO  # noqa: E402
SEL_JSON = REPO / "experiments/fig11_probe_controls/probe_selectivity_controls/probe_selectivity_controls.json"
LAY_JSON = REPO / "experiments/fig11_probe_controls/probe_layerwise_emergence/layerwise_r2.json"

# Applied via rc_context inside make_figure, NOT module-wide, so the settings
# don't leak into later figures in a shared kernel.
# Authored at 7.0in and printed at \textwidth = 5.5in (x0.79), so 10pt here
# prints at ~8pt, the paper's caption size.
RC = {
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10.5,
    "legend.fontsize": 9,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
}
TAG_FS = 11   # panel letters


def err(x):
    return np.clip(1.0 - np.asarray(x, dtype=float), 1e-6, None)


def band(e):
    """(mean, min, max) over the seed axis (axis 0) of a stack of error curves."""
    e = np.asarray(e, dtype=float)
    return e.mean(0), e.min(0), e.max(0)


def plot_band(ax, x, e, color, ls, lw, label=None, **kw):
    """Mean line over seeds with a min-max shaded band."""
    m, lo, hi = band(e)
    ax.fill_between(x, lo, hi, color=color, alpha=0.2, lw=0, zorder=1.5)
    ax.plot(x, m, ls, color=color, lw=lw, label=label, zorder=2.5, **kw)


def style_axis(ax):
    ax.set_xscale("symlog", linthresh=1.0, linscale=0.4)
    ax.set_yscale("log")
    ax.set_xlim(-0.3, 6.5e4)
    ax.grid(True, which="major", alpha=0.3)


def make_figure(out_dir: Path):
    with plt.rc_context(RC):
        _make_figure(Path(out_dir) / "figure_probe_controls")


def _make_figure(OUT: Path):
    sel = json.loads(SEL_JSON.read_text())
    lay = json.loads(LAY_JSON.read_text())

    fig = plt.figure(figsize=(7.0, 5.7))
    gs = fig.add_gridspec(2, 4, hspace=0.55, wspace=0.32,
                          left=0.095, right=0.99, top=0.945, bottom=0.275)

    # ------------------- top row: selectivity -------------------
    # The stored keys name fractal2-as-$T$ "A"; the paper names the mess3-as-T
    # arrangement A -- it is the one the main text shows -- so the keys are
    # remapped on the way out and reordered so A leads. Must agree with
    # SETUP_LETTERS in plotting/feedback_msp.py.
    DISPLAY = {"A": "B", "B": "A"}
    KEY_ORDER = ["B", "A"]          # stored keys, in display order
    SETUP_TITLES = [
        "setup A:  $T$ = mess3,  $U$ = fractal2",
        "setup B:  $T$ = fractal2,  $U$ = mess3",
    ]
    CURVES = [
        ("true_f0", "true belief $T$", "tab:blue", "-"),
        ("true_f1", "true belief $U$", "tab:orange", "-"),
        ("shift", "time-shifted ($t-2$)", "tab:purple", "-."),
        ("shuffle", "shuffled sequences", "tab:red", ":"),
    ]
    sel_axes = []
    sel_setups = sel.get("setups") or [{"seeds": [r]} for r in sel["runs"]]
    # stored order is [A, B]; show them in display order
    sel_setups = [sel_setups[1], sel_setups[0]]
    for i, setup in enumerate(sel_setups):
        ax = fig.add_subplot(gs[0, 2 * i:2 * i + 2])
        sel_axes.append(ax)
        seeds = setup["seeds"]
        # steps present in every seed (they agree in practice)
        common = sorted(set.intersection(*(set(s["steps"]) for s in seeds)))
        steps = np.asarray(common, dtype=float)
        for key, label, color, ls in CURVES:
            e = [err([dict(zip(s["steps"], s["r2"][key]))[c] for c in common])
                 for s in seeds]
            plot_band(ax, steps, e, color, ls, 1.8, label=label)
        style_axis(ax)
        ax.set_ylim(1e-4, 3.0)
        ax.set_title(SETUP_TITLES[i])
        ax.set_xlabel("optimizer step")
        if i == 0:
            ax.set_ylabel("probe error  $1-R^2$")
        ax.text(0.02, 0.965, f"({'ab'[i]})", transform=ax.transAxes,
                fontweight="bold", fontsize=TAG_FS, va="top")
    handles, labels = sel_axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.10))

    # ------------------- bottom row: layer-wise -------------------
    steps_l = np.asarray(lay["steps"], dtype=float)
    sites = lay["site_labels"]
    cmap = plt.get_cmap("viridis")
    site_colors = [cmap(x) for x in np.linspace(0.0, 0.88, len(sites))]
    PANELS = [(k, f) for k in KEY_ORDER for f in (0, 1)]
    for j, (setup, f) in enumerate(PANELS):
        ax = fig.add_subplot(gs[1, j])
        d = lay["setups"][setup]
        # [seed, step, site, factor]; fall back to the v1 single-seed array
        r2 = np.asarray(d["r2_seeds"]) if "r2_seeds" in d \
            else np.asarray(d["r2"])[None]
        for s in range(len(sites)):
            ls = "--" if sites[s] == "ln final" else "-"
            plot_band(ax, steps_l, err(r2[:, :, s, f]), site_colors[s], ls,
                      1.7, label=sites[s], marker="o", ms=2.6)
        style_axis(ax)
        # four ticks: six symlog decades do not fit a 1.4in panel at 8.5pt
        ax.set_xticks([0, 1e0, 1e2, 1e4])
        ax.set_ylim(6e-5, 0.6)
        ax.set_xlabel("optimizer step")
        if j == 0:
            ax.set_ylabel("probe error  $1-R^2$")
        else:
            ax.set_yticklabels([])
        ax.set_title(f"{DISPLAY[setup]} $\\cdot$ " + d["factor_labels"][f])
        ax.text(0.05, 0.045, f"({'cdef'[j]})", transform=ax.transAxes,
                fontweight="bold", fontsize=TAG_FS, va="bottom")
    handles, labels = fig.axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False,
               bbox_to_anchor=(0.5, 0.0),
               title="residual-stream site (depth)", title_fontsize=9)

    fig.savefig(OUT.with_suffix(".pdf"))
    fig.savefig(OUT.with_suffix(".png"), dpi=180)
    plt.close(fig)
    print("wrote", OUT.with_suffix(".pdf"), "and .png")


