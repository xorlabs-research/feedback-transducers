"""The paper's dimensional-advantage figure, emitted as ONE pdf.

  figure_dimension_full.pdf    the whole figure the paper includes:
                               (a,b) the two halves of the social-network
                               diagram on top, (c,d,e) the data row below

Assembled by analysis/figure_dimension_full.tex, a `standalone` wrapper this
module runs at the end. The assembly stays in LaTeX because the diagram is TikZ
(analysis/social_network.tikz: \\snHubPanel and \\snLoopPanel) that matplotlib
could only embed as a raster.

This module draws the data row that goes into it:

  figure_dimension_row.pdf     (c) telephone CEV, 3 mess3 sub-agents
                               (d) effective rank vs N
                               (e) coarse-graining CEV

authored at 6.6in wide, i.e. ~0.83x of the \\textwidth it prints at. The
appendix keeps one separate cell, authored at the size it prints at:

  figure_cell_tel_cev_f2.pdf   telephone CEV, 5 fractal2 sub-agents

Panels are drawn by the SAME functions as the standalone figures
(telephone_cev.draw_cev_panel, effective_rank_vs_n.draw_rank_panel,
coarse_graining_figure.panel_cev), so the versions cannot drift apart.

Ported verbatim from the frozen `figure_dimension_cells.py`.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

from . import telephone_cev as cs
from . import coarse_graining_figure as cg
from .telephone_combined import rank_points, THRESHOLDS
from .effective_rank_vs_n import draw_rank_panel
from .lib.runs import REPO

CG_JSON = REPO / "analysis" / "coarse_graining_trained.json"
TEX = REPO / "analysis" / "figure_dimension_full.tex"
TIKZ = REPO / "analysis" / "social_network.tikz"
# (a,b) as the paper prints them: the co-authors' hand-drawn export, cropped by
# its wrapper. Without it the wrapper above draws them from the TikZ.
TEX_MANUAL = REPO / "analysis" / "figure_dimension_full_manual.tex"
AB_MANUAL = REPO / "analysis" / "social_network_ab_manual.pdf"

# the appendix cell, authored at the size it prints at (0.36\textwidth). It is
# wider than the row's panels: at these font sizes a 1.56in canvas leaves the
# title and the legend fighting the curves for the same inch.
CELL = (2.00, 1.65)
MARGINS = dict(left=0.200, right=0.983, top=0.885, bottom=0.206)

# Sizes as they should PRINT. The row figure divides by its own reduction.
FS = 8.0         # panel titles + their tags; shared with feedback_msp's
                 # _tagged_title, so panel letters match across figures
FS_LAB = 8.0     # axis labels
FS_TICK = 6.5    # tick labels
FS_LEG = 6.0     # legend entries

ROW = (6.6, 2.05)            # inches; included at \textwidth -> ~0.833x
ROW_SCALE = 5.5 / ROW[0]     # \textwidth over the authored width
# Panel letters print at 8pt here, in plotting/feedback_msp.py (Fig. 2) and in
# figure_dimension_full.tex (panels a,b), so they match across the figures.
TAG_FS = 8.0 / ROW_SCALE
# left margin holds a whole rotated "CEV" plus its tick labels -- the combined
# pdf is cropped to this canvas, so anything tight against the edge loses a
# sliver. right leaves room for the widest panel's last x tick label.
ROW_MARGINS = dict(left=0.088, right=0.986, top=0.875, bottom=0.185,
                   wspace=0.38)
# label every 4th rank: every other one is ten labels across a panel this size
XTICK_STEP = 4


def _row(pt):
    """A print size, in the row figure's authored points."""
    return pt / ROW_SCALE


def _title(ax, tag, text, fs, tag_dx=-2.0):
    """Centred panel title with its bold tag out at the axes' LEFT EDGE, which
    is where the (a,b) letters sit in the diagram row above -- packing the tag
    against the title instead put it at a different offset in every panel,
    since the offset then depends on how wide the title is.

    tag_dx is points from that edge, negative to hang outside it; the margin
    there is the y label's, and the label sits well below the title band.
    """
    ax.set_title(text, fontsize=fs, pad=3)
    ax.annotate(tag, xy=(0.0, 1.0), xycoords="axes fraction",
                xytext=(tag_dx, 3), textcoords="offset points",
                ha="right", va="bottom", fontsize=TAG_FS, fontweight="bold")


def _cev_panel(ax, p, tag, title, fs, fs_lab, fs_tick, fs_leg):
    cs.draw_cev_panel(ax, p, title, fs=fs, compact=True, xtick_step=XTICK_STEP)
    _title(ax, tag, title, fs)
    ax.set_ylabel("CEV", fontsize=fs_lab)
    ax.xaxis.label.set_fontsize(fs_lab)
    ax.tick_params(labelsize=fs_tick)
    for t in ax.get_legend().get_texts():
        t.set_fontsize(fs_leg)


def _rank_panel(ax, pts, fs, fs_lab, fs_tick, fs_leg):
    draw_rank_panel(ax, pts, THRESHOLDS, title=False, fs=fs, compact=True,
                    minor_grid=False)
    _title(ax, "(d)", "effective rank vs. $N$", fs)
    ax.set_ylabel("$n_c$ (log)", fontsize=fs_lab, labelpad=1)
    ax.xaxis.label.set_fontsize(fs_lab)
    # draw_rank_panel sets both legends (style / threshold colour) itself;
    # only their font size is restyled here
    for leg in [a for a in ax.get_children()
                if isinstance(a, matplotlib.legend.Legend)]:
        for t in leg.get_texts():
            t.set_fontsize(fs_leg)
    ax.tick_params(labelsize=fs_tick)


def _coarse_panel(ax, fs, fs_lab, fs_tick, fs_leg):
    reps, by = cg._load(CG_JSON)
    cg.panel_cev(ax, reps, by, short_labels=True)
    _title(ax, "(e)", "coarse-graining", fs)
    ax.set_ylabel("CEV", fontsize=fs_lab)
    ax.set_xlabel("rank $r$", fontsize=fs_lab)
    ax.tick_params(labelsize=fs_tick)
    leg = ax.get_legend()
    if leg is not None:
        for t in leg.get_texts():
            t.set_fontsize(fs_leg)


def build_full_figure(out_dir: Path) -> None:
    """Run the standalone wrapper that puts the diagram halves and the data row
    on one page. Aux files go to a temp dir, so only the pdf lands in out_dir.
    The wrapper, its (a,b) source and the already-drawn row pdf are staged
    into the temp dir so the wrapper's relative paths resolve."""
    out_dir = Path(out_dir)
    tex, ab = (TEX_MANUAL, AB_MANUAL) if AB_MANUAL.exists() else (TEX, TIKZ)
    if shutil.which("pdflatex") is None:
        print("pdflatex not on PATH; skipping figure_dimension_full.pdf")
        return
    with tempfile.TemporaryDirectory() as tmp:
        for src in (tex, ab, out_dir / "figure_dimension_row.pdf"):
            shutil.copy(src, tmp)
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode",
                            "-halt-on-error", f"-output-directory={tmp}",
                            tex.name],
                           cwd=tmp, capture_output=True, text=True)
        out = Path(tmp) / tex.with_suffix(".pdf").name
        if r.returncode != 0 or not out.exists():
            print(r.stdout[-2000:])
            raise SystemExit(f"{tex.name} failed to build")
        shutil.copy(out, out_dir / "figure_dimension_full.pdf")
    print("wrote figure_dimension_full.pdf")


def make_figures(out_dir: Path, assemble: bool = True) -> None:
    """assemble=False skips the pdflatex step that composes the data row with
    the conceptual TikZ diagram panels into figure_dimension_full.pdf -- the
    data subfigures (row + appendix cell) are drawn either way."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("collecting telephone CEV panels (5 seeds each) ...", flush=True)
    panels = [cs.collect_panel(rds) for rds in cs.PANELS]
    print("collecting rank-vs-N points ...", flush=True)
    pts = rank_points()

    # --- the paper's bottom row, one pdf ---------------------------------
    fig, axes = plt.subplots(1, 3, figsize=ROW)
    fig.subplots_adjust(**ROW_MARGINS)
    sizes = [_row(FS), _row(FS_LAB), _row(FS_TICK), _row(FS_LEG)]
    _cev_panel(axes[0], panels[1], "(c)", "3 mess3 sub-agents", *sizes)
    _rank_panel(axes[1], pts, *sizes)
    _coarse_panel(axes[2], *sizes)
    fig.savefig(out_dir / "figure_dimension_row.png", dpi=200)
    fig.savefig(out_dir / "figure_dimension_row.pdf")
    plt.close(fig)
    print("wrote figure_dimension_row.pdf")

    # --- the appendix's fractal2 cell, at its printed size ---------------
    fig, ax = plt.subplots(figsize=CELL)
    cs.draw_cev_panel(ax, panels[0], "5 fractal2 sub-agents", fs=FS,
                      compact=True, xtick_step=XTICK_STEP)
    ax.set_title("5 fractal2 sub-agents", fontsize=FS, pad=3)
    ax.set_ylabel("CEV", fontsize=FS_LAB)
    ax.xaxis.label.set_fontsize(FS_LAB)
    ax.tick_params(labelsize=FS_TICK)
    for t in ax.get_legend().get_texts():
        t.set_fontsize(FS_LEG)
    fig.subplots_adjust(**MARGINS)
    fig.savefig(out_dir / "figure_cell_tel_cev_f2.png", dpi=200)
    fig.savefig(out_dir / "figure_cell_tel_cev_f2.pdf")
    plt.close(fig)
    print("wrote figure_cell_tel_cev_f2.pdf")

    if assemble:
        build_full_figure(out_dir)
