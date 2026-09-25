"""Feedback MSP reconstruction + belief-patching figures (the paper's
feedback-loop figures, main text and appendix).

`pair` selects which role-swapped pair of 50k canonical runs the figure is
built from ("sns" or "fractal2"; the paper's main text uses fractal2). One make_figures() call
emits, for stem = figure_feedback_msp[_fractal2]:

  <stem>.{png,pdf}                       both setups, MSP + KL column
  <stem>_setupA.{png,pdf}                single-row main-text layout
  <stem>_setup{A,B}_timeline.{png,pdf}   per-setup training timelines
  <stem>_timeline.{png,pdf}              both-setup timeline
  figure_steering_setupA.{png,pdf}       standalone belief-patching bars

The KL column reads analysis/steering_kl_controls_paper.json through
plotting.steering_kl_controls.
"""
from __future__ import annotations

import math
import string
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea

from . import steering_kl_controls as kc
from .lib.feedback_runs import (
    CKPT_FINAL, SEED, acts_at_last_resid_post, u_action_size,
    color_from_marginal, emission_marginal, fit_predict_linear, load_cfg_proc,
    load_model, pca_basis, project, sample_batch, stack_beliefs,
)
from .lib.runs import resolve_run

# `pair` selects which coupled pair the figure is built from. Both are
# role-swapped pairs of 50k canonical runs, so the panel layout is identical.
#   sns                the original Fig. "feedback_msp": mess3 <-> sns
#   fractal2           the paper's pair: mess3 <-> fractal2, whose factors
#                      both steer perfectly -- the visual counterpart to the
#                      steering result, with no nonunifilar-source confound
PAIRS = {
    "sns": {
        "runs": [("A", "feedback_sns_mess3_canonical_50k"),
                 ("B", "feedback_mess3_sns_canonical_50k")],
        "col_titles": ["agent $T$ = SNS,  agent $U$ = mess3",
                       "agent $T$ = mess3,  agent $U$ = SNS"],
        # T = the non-delayed factor (factor 0), U = the action-delayed one
        "row_labels": [["agent $T$ (SNS)", "agent $U$ (mess3)"],
                       ["agent $T$ (mess3)", "agent $U$ (SNS)"]],
        "stem": "figure_feedback_msp",
    },
    "fractal2": {
        "runs": [("A", "feedback_fractal2_mess3_canonical_50k"),
                 ("B", "feedback_mess3_fractal2_canonical_50k")],
        "col_titles": ["$T$ = fractal2,  $U$ = mess3",
                       "$T$ = mess3,  $U$ = fractal2"],
        "row_labels": [["$T$ (fractal2)", "$U$ (mess3)"],
                       ["$T$ (mess3)", "$U$ (fractal2)"]],
        "stem": "figure_feedback_msp_fractal2",
        # steering_kl_controls_paper.py setup keys for (setup A, setup B)
        "kl_setups": ["loop_fractal2_mess3", "loop_mess3_fractal2"],
        # setup shown in the main text; the other one lives in the appendix
        # figure that keeps both (see make_msp_figure's `setups`)
        "main_setup": 1,
        "orient": True,          # data-driven rotation + legend (see _orient)
        "bar_style": (0.15, 0.14),   # thinner/fainter: fractal2's barcode
                                     # saturated at the default 0.35/0.28
    },
}
# draw_controls_panel pins the KL axis's lower limit here; the row layout needs
# it to size a shortened axis in decades.
# Panel letters: 9.6pt authored is 8pt printed at this figure's 0.833x
# reduction. plotting/dimension_cells.py and figure_dimension_full.tex use the
# same printed size, so the letters match across the paper's figures.
TAG_FS = 9.6

# Display letter per results index. Both pairs are ordered [T = the 2-state
# process, T = mess3], and the paper names the mess3-as-T arrangement A: it is
# the one the main text shows, so it leads in the appendix. Flip this to
# ("A", "B") to go back to naming them in run order.
SETUP_LETTERS = ("B", "A")
# appendix figures put A first, whichever run that is
SETUP_ORDER = tuple(sorted(range(2), key=lambda i: SETUP_LETTERS[i]))

KL_YMIN = 2e-5
# decades a KL panel spanned when it was drawn at full cell height; the row
# layout keeps this many decades per inch while showing fewer of them.
KL_DECADES_FULL = 6.48

# 4 log-spaced snapshots (decade spacing).
TIMELINE_STEPS = [0, 500, 5000, 50000]


# Colour directions get placed at these angles (degrees, counter-clockwise).
# 45/165/285 puts an edge of the colour triangle toward the bottom-left corner,
# leaving that corner -- where _color_legend anchors -- outside the point cloud.
TARGET_ANGLES = np.array([45.0, 165.0, 285.0])


def _colour_dirs(coords, weights, top_frac=0.10):
    """Unit direction from the cloud centre toward each colour's extreme, in
    projected coordinates. Uses the top decile of each colour weight rather than
    the simplex vertices, so it works both for belief-coloured panels and for
    emission-marginal-coloured ones (whose colour is not the plotted quantity)."""
    c0 = coords.mean(0)
    dirs = []
    for k in range(weights.shape[1]):
        w = weights[:, k]
        thr = np.quantile(w, 1.0 - top_frac)
        sel = w >= thr
        d = coords[sel].mean(0) - c0
        n = np.linalg.norm(d)
        dirs.append(d / n if n > 0 else np.array([1.0, 0.0]))
    return np.array(dirs)


def _orient(rows, mu, basis, weights):
    """Rotate (and if needed reflect) a 2-D PCA basis so the colour directions
    land on TARGET_ANGLES. Returns the new basis and the resulting directions."""
    coords = project(rows, mu, basis)
    dirs = _colour_dirs(coords, weights)
    # enforce counter-clockwise colour order, so a rotation alone can hit the
    # targets; a mirrored projection needs one axis flipped first
    cross = dirs[0, 0] * dirs[1, 1] - dirs[0, 1] * dirs[1, 0]
    if cross < 0:
        basis = basis.copy()
        basis[:, 1] *= -1.0
        coords = project(rows, mu, basis)
        dirs = _colour_dirs(coords, weights)
    phi = np.deg2rad(TARGET_ANGLES[0]) - np.arctan2(dirs[0, 1], dirs[0, 0])
    c, s = np.cos(phi), np.sin(phi)
    basis = basis @ np.array([[c, s], [-s, c]])
    return basis, _colour_dirs(project(rows, mu, basis), weights)


def collect(run_dir: Path, orient: bool = False) -> dict:
    """Per-factor truth MSP + per-checkpoint probe reconstructions (truth PCA basis).

    orient: rotate 3-colour panels so the colour triangle's edge faces the
    bottom-left legend corner. Off by default, which keeps the published sns
    figure identical."""
    cfg, process = load_cfg_proc(run_dir)
    inputs, beliefs = sample_batch(cfg, process)
    b_list = stack_beliefs(beliefs)
    action_size = u_action_size(cfg)

    B = b_list[0].shape[0]
    T = b_list[0].shape[1] - 1  # skip BOS
    rng = np.random.default_rng(SEED + 1)
    idx = np.arange(B); rng.shuffle(idx)
    cut = int(0.8 * B)
    tr_mask = np.zeros(B, dtype=bool); tr_mask[idx[:cut]] = True
    is_tr = tr_mask[np.repeat(np.arange(B), T)]
    is_te = ~is_tr

    factors = []
    for fi, b in enumerate(b_list):
        bf = b[:, 1:]
        D = bf.shape[-1]
        b_flat = bf.reshape(-1, D)
        if fi == 0:
            colors = color_from_marginal(b_flat)  # agent T: color by the belief itself
            n_color, kind = D, "state"
        else:
            p_emit = emission_marginal(bf, action_size).reshape(-1, action_size)
            colors = color_from_marginal(p_emit)
            n_color, kind = action_size, "emit"
        mu_b, basis_b = pca_basis(b_flat, k=2)
        # For 3-colour panels, orient the projection from the data: rotate so an
        # EDGE of the colour triangle faces the bottom-left corner (colour
        # directions at 45/165/285 deg), which is where the legend inset sits --
        # a vertex there would sit under it. The same directions are handed to
        # the legend, so its dots cannot disagree with the plot.
        col_dirs = None
        if orient and n_color == 3:
            weights = b_flat if fi == 0 else p_emit
            basis_b, col_dirs = _orient(b_flat, mu_b, basis_b, weights)
        coords_theory = project(b_flat, mu_b, basis_b)
        # A 2-state belief simplex is 1-D: PC2 is numerical noise. Render such
        # factors as a BARCODE (one vertical line per sample at its belief
        # coordinate) instead of inventing a second dimension.
        factors.append(dict(dim=D, b_flat=b_flat, colors=colors,
                            mu=mu_b, basis=basis_b, barcode=(D == 2),
                            n_color=n_color, kind=kind, col_dirs=col_dirs,
                            coords_theory=coords_theory,
                            per_step={}))

    for step in sorted(set(TIMELINE_STEPS) | {CKPT_FINAL}):
        ckpt = run_dir / f"checkpoints/{step}/model.pt"
        if not ckpt.exists():
            print(f"    missing ckpt {step}")
            continue
        m = load_model(cfg, ckpt)
        a = acts_at_last_resid_post(m, inputs)[:, 1:]
        a = a.reshape(-1, a.shape[-1])
        for fi, fac in enumerate(factors):
            pred = fit_predict_linear(a[is_tr], fac["b_flat"][is_tr], a)
            ss_res = float(np.sum((fac["b_flat"][is_te] - pred[is_te]) ** 2))
            ss_tot = float(np.sum((fac["b_flat"][is_te] - fac["b_flat"][is_te].mean(0)) ** 2))
            r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
            fac["per_step"][step] = (project(pred, fac["mu"], fac["basis"]), r2)
            print(f"    step {step} factor {fi} (dim {fac['dim']}): held-out R2 = {r2:.4f}")

    return dict(action_size=action_size, factors=factors)


def _scatter(ax, xy, colors, lims, s=3.5, alpha=0.85, barcode=False,
             bar_lw=0.35, bar_alpha=0.28):
    if barcode:
        # Half-height bars, vertically centered within the panel. With ~7000
        # samples over a 1-D belief set, lw/alpha decide whether the Cantor gaps
        # are visible or the strip saturates into a solid gradient; calibrated at
        # the panel's real width, since that is what sets the overlap.
        ax.vlines(xy[:, 0], 0.25, 0.75, colors=colors, lw=bar_lw,
                  alpha=bar_alpha)
        ax.set_xlim(lims[0]); ax.set_ylim(0.0, 1.0)
    else:
        ax.scatter(xy[:, 0], xy[:, 1], c=colors, s=s, lw=0, alpha=alpha)
        ax.set_xlim(lims[0]); ax.set_ylim(lims[1])
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def _color_legend(ax, n_color, kind, dirs=None, fs=6.0):
    """Color-encoding legend, anchored bottom-left of the subplot. `kind`=
    'state' (agent-T panels, colored by the belief itself) or 'emit' (agent-U panels,
    colored by the emission marginal). The 2-category (sns) legend is a flat
    horizontal bar; the 3-category one uses RGB corners."""
    if n_color == 2:
        # Flat horizontal colorbar (~20% of the square legend's height).
        inset = ax.inset_axes([0.03, 0.13, 0.30, 0.05])
        cmap = LinearSegmentedColormap.from_list("br", [(0, 0, 1), (1, 0, 0)])
        grad = np.linspace(0, 1, 100)[None, :].repeat(8, 0)
        inset.imshow(grad, cmap=cmap, aspect="auto", extent=(0, 1, 0, 1))
        inset.set_yticks([])
        inset.set_xticks([0, 1]); inset.set_xticklabels(["0", "1"], fontsize=fs - 0.5)
        inset.tick_params(length=1.5, pad=1)
        lbl = "belief $P(s{=}0)$" if kind == "state" else "$P(\\mathrm{emit}{=}0)$"
    else:
        inset = ax.inset_axes([0.02, 0.10, 0.24, 0.24])
        inset.patch.set_facecolor("white"); inset.patch.set_alpha(0.9)
        inset.set_xticks([]); inset.set_yticks([])
        # dot layout mirrors the panel's actual colour directions, so the legend
        # reads as a miniature of the cloud rather than a fixed triangle
        if dirs is None:
            pos = [(0.5, 0.88), (0.07, 0.13), (0.93, 0.13)]
        else:
            pos = [(0.5 + 0.36 * d[0], 0.5 + 0.36 * d[1]) for d in dirs]
        for v, col, l in zip(pos,
                             [(1, 0, 0), (0, 1, 0), (0, 0, 1)], ["0", "1", "2"]):
            inset.scatter([v[0]], [v[1]], c=[col], s=13, edgecolors="black", lw=0.25)
            inset.annotate(l, (v[0], v[1]), xytext=(0, 3), textcoords="offset points",
                           ha="center", fontsize=fs - 0.5)
        inset.set_xlim(0, 1); inset.set_ylim(0, 1)
        lbl = "RGB $=$ belief" if kind == "state" else "RGB $= P(\\mathrm{emit})$"
    for sp in inset.spines.values():
        sp.set_edgecolor("0.6"); sp.set_linewidth(0.4)
    inset.set_xlabel(lbl, fontsize=fs, labelpad=1)


def _lims(coords, pad=0.06):
    xp = (coords[:, 0].max() - coords[:, 0].min()) * pad
    yp = (coords[:, 1].max() - coords[:, 1].min()) * pad
    xp = xp if xp > 0 else 0.05
    yp = yp if yp > 0 else 0.05
    return ((coords[:, 0].min() - xp, coords[:, 0].max() + xp),
            (coords[:, 1].min() - yp, coords[:, 1].max() + yp))


def _row_label(ax, text, kw):
    """Rotated label to the left of a panel. `kw` (from make_msp_figure) places
    it in offset points when the panel tags share the same margin; None keeps
    the original axes-fraction placement."""
    if kw is None:
        ax.text(-0.06, 0.5, text, transform=ax.transAxes, rotation=90,
                va="center", ha="right", fontsize=8.5)
    else:
        ax.annotate(text, xy=(0.0, 0.5), rotation=90, va="center",
                    ha="right", fontsize=8.5, **kw)


def _kl_panel(ax, per_setup, key, *, ymax=10.0, gap_y=1.9, gap_axes_y=None,
              gap_oneline=False, show_key=True, key_y0=0.72, key_dy=0.095,
              key_cols=1, key_row=False, label_every=1, fs_lab=7.5,
              fs_tick=7, fs_gap=None, fs_ylab=9):
    """The joint-KL steering bars for one setup, one bar group per PATCHED
    belief. Shared by both layouts so they cannot drift apart; the caller owns
    the title, the panel tag and the x tick labels."""
    fs_gap = fs_tick if fs_gap is None else fs_gap
    rows_all = per_setup[key][0]
    # synthesize a pseudo-setup per edited factor so the shared drawer can
    # group by which belief was patched
    facs = sorted({e["factor"] for e in rows_all})
    sub, order, gnames = {}, [], []
    for f in facs:
        k = f"{key}::f{f}"
        sub[k] = ([e for e in rows_all if e["factor"] == f], per_setup[key][1])
        order.append(k)
        gnames.append(f"patch ${'TU'[f] if f < 2 else f}$")
    kc.draw_controls_panel(ax, sub, fs_lab=fs_lab, fs_tick=fs_tick,
                           legend=False, order=order, group_labels=gnames,
                           ymax=ymax, gap_y=None)
    for gi, k in enumerate(order):
        rr = sub[k][0]
        # per-seed gap, so the spread quoted is across seeds; the group mean is
        # the ratio of means, which is what the bars show
        gaps = np.array([1 - e["measured"] / e["control_null"] for e in rr])
        gap = 1 - (np.mean([e["measured"] for e in rr])
                   / np.mean([e["control_null"] for e in rr]))
        lab = f"{100 * gap:.1f}%"
        if len(gaps) > 1:
            sep = "$\\pm$" if gap_oneline else "\n$\\pm$"
            lab += f"{sep}{100 * gaps.std(ddof=1):.1f}"
        # hangs downward from its anchor: growing upward, it runs into whatever
        # sits above the bars. gap_axes_y pins it to a fraction of the panel
        # instead of a KL value, for panels too short to place it by value.
        tr = ax.get_xaxis_transform() if gap_axes_y is not None else ax.transData
        ax.text(gi, gap_y if gap_axes_y is None else gap_axes_y, lab,
                transform=tr, ha="center", va="top", fontsize=fs_gap,
                fontweight="bold", color="0.2", linespacing=0.95)
    if show_key:
        # the four bars need naming once; upper-left is clear because the floor
        # and edit bars are ~1e-4 and ~3e-3
        n_q = len(kc.QUANTITIES)
        for j, (_, _, colour, _) in enumerate(kc.QUANTITIES):
            # key_row spreads the entries across one centred line -- for panels
            # whose bars leave a band free at the top but no free corner
            x, ha = (((j + 0.5) / n_q, "center") if key_row
                     else (0.03 + 0.49 * (j % key_cols), "left"))
            ax.text(x, key_y0 - (0.0 if key_row else key_dy * (j // key_cols)),
                    "\u25a0 " + kc.BAR_SHORT[j],
                    transform=ax.transAxes, fontsize=6, color=colour,
                    ha=ha, va="top")
    if label_every > 1:
        # label every nth decade: on a short axis a label per decade is a
        # denser column of text than the bars it annotates
        lo, hi = ax.get_ylim()
        decades = [10.0 ** e for e in
                   range(math.ceil(math.log10(lo)), math.floor(math.log10(hi)) + 1)]
        ax.set_yticks(decades)
        ax.set_yticklabels([f"$10^{{{round(math.log10(d))}}}$"
                            if i % label_every == 0 else ""
                            for i, d in enumerate(decades)])
    ax.set_ylabel("KL  [nats]", fontsize=fs_ylab)
    # unboxed, like the MSP panels: keep only the axes the bars are measured
    # against
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.6)
        ax.spines[side].set_color("0.3")


def make_msp_figure(results, out_path: Path, kl_setups=None,
                    bar_style=(0.35, 0.28), setups=(0, 1)):
    """MSP panels for the setups named in `setups` (indices into `results`):
    2 rows (truth / reconstruction) x 2 cols per setup (its two factors), plus
    an optional trailing column with the joint-KL steering bars (same panel
    drawer as figure_steering_combined_kl). With two setups the KL column
    carries one panel per setup; with one it spans both rows."""
    setups = list(setups)
    nset = len(setups)
    n_msp = 2 * nset
    has_kl = kl_setups is not None
    # 0.12 spacer keeps the KL y axis off the last MSP panel; a lone KL panel
    # is wider because it spans both rows
    widths = ([1.0] * n_msp
              + ([0.12, 1.10 if nset > 1 else 1.35] if has_kl else []))
    ncol = len(widths)
    fig_w = (8.15 if nset > 1 else 5.42) if has_kl else 1.75 * n_msp
    # side margins fixed in INCHES: the rotated row labels and the outside
    # panel tags need the same room however many setups are drawn
    left = 0.062 * 8.15 / fig_w if has_kl else 0.055
    right = 1.0 - 0.005 * 8.15 / fig_w if has_kl else 0.995
    fig = plt.figure(figsize=(fig_w, 3.9))
    gs = fig.add_gridspec(2, ncol, hspace=0.08,
                          wspace=0.34 if has_kl else 0.08,
                          width_ratios=widths,
                          left=left, right=right, top=0.845,
                          bottom=0.14 if has_kl else 0.02)
    # panels are tagged in reading order: top row, bottom row, then KL
    tags = [f"({c})" for c in string.ascii_lowercase]
    labels = [tags[:n_msp], tags[n_msp:2 * n_msp]]
    kl_tags = tags[2 * n_msp:2 * n_msp + nset]
    panels = [(setup, fi) for setup in setups for fi in range(2)]
    outside = has_kl      # labels left of the y axis
    lab_kw = (dict(x=-0.04, y=1.0, ha="right", va="top") if outside
              else dict(x=0.02, y=0.97, ha="left", va="top"))
    # With the tags outside, they share the left margin with the rotated row
    # labels. Offsets there are in POINTS, not axes fractions: the row label
    # has to clear the tag by the tag's own rendered width, which does not
    # shrink when the figure carries fewer setups.
    rowlab = (dict(xytext=(-17, 0), textcoords="offset points",
                   xycoords="axes fraction")
              if outside else None)
    top_axes, bot_axes, kl_axes, kl_labels = [], [], [], []
    for col, (setup, fi) in enumerate(panels):
        r = results[setup]
        fac = r["factors"][fi]
        final_xy = fac["per_step"][CKPT_FINAL][0]
        lims = _lims(np.vstack([fac["coords_theory"], final_xy]))
        ax = fig.add_subplot(gs[0, col])
        _scatter(ax, fac["coords_theory"], fac["colors"], lims, s=1.6,
                 barcode=fac["barcode"], bar_lw=bar_style[0],
                 bar_alpha=bar_style[1])
        # theoretical belief-geometry dimension = simplex dim = (#states - 1)
        ax.set_title(f"{r['row_labels'][fi]}, $d\\!=\\!{fac['dim'] - 1}$",
                     fontsize=9)
        top_axes.append(ax)
        ax.text(lab_kw["x"], lab_kw["y"], labels[0][col], transform=ax.transAxes,
                fontsize=10.5, fontweight="bold", ha=lab_kw["ha"], va=lab_kw["va"])
        _color_legend(ax, fac["n_color"], fac["kind"], fac.get("col_dirs"))
        if col == 0:
            _row_label(ax, "Bayes-optimal MSP", rowlab)
        ax = fig.add_subplot(gs[1, col])
        bot_axes.append(ax)
        _scatter(ax, final_xy, fac["colors"], lims, s=1.6,
                 barcode=fac["barcode"], bar_lw=bar_style[0],
                 bar_alpha=bar_style[1])
        ax.text(lab_kw["x"], lab_kw["y"], labels[1][col], transform=ax.transAxes,
                fontsize=10.5, fontweight="bold", ha=lab_kw["ha"], va=lab_kw["va"])
        ax.text(0.03, 0.04, f"$R^2={fac['per_step'][CKPT_FINAL][1]:.3f}$",
                transform=ax.transAxes, fontsize=7.5, ha="left", va="bottom",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="0.6",
                          lw=0.4, alpha=0.9))
        if col == 0:
            _row_label(ax, "transformer\nreconstruction", rowlab)
    # optional KL column: per setup, one bar group per PATCHED belief
    if has_kl:
        per_setup = kc.load()
        # a lone panel spans both rows, so the key goes right to the top and
        # the axis gets an extra decade of headroom: otherwise the key and the
        # gap-closed annotations both want the strip just above the bars
        key_y0, key_dy = (0.72, 0.095) if nset > 1 else (0.99, 0.05)
        kl_ymax = 10.0 if nset > 1 else 60.0
        for i, setup in enumerate(setups):
            ax = fig.add_subplot(gs[:, ncol - 1] if nset == 1
                                 else gs[i, ncol - 1])
            kl_axes.append(ax)
            _kl_panel(ax, per_setup, kl_setups[setup], ymax=kl_ymax,
                      show_key=(i == 0), key_y0=key_y0, key_dy=key_dy)
            if i == 0:
                # one title for the column, and (when the column is split per
                # setup) no bar labels on the top panel: a per-row title and
                # rotated labels collide across the small inter-row gap
                ax.set_title("Belief patching", fontsize=9)
                if nset > 1:
                    # group labels only on the bottom panel: on the top one
                    # they land on the lower panel's setup indicator
                    ax.set_xticklabels([])
            kl_labels.append(
                ax.text(-0.26, 1.0, kl_tags[i], transform=ax.transAxes,
                        fontsize=10.5, fontweight="bold", ha="right", va="top"))
            if nset > 1:
                # which setup this panel belongs to; with a single setup the
                # figure's one group title above already says it
                ax.text(0.5, 0.995, f"setup {SETUP_LETTERS[setup]}",
                        transform=ax.transAxes, fontsize=9, fontweight="bold",
                        ha="center", va="top", color="0.15")

    # thin grey rule between the MSP block and the KL column, placed to the
    # LEFT of the (i)/(j) labels: those hang outside their axes, so the middle
    # of the axes-to-axes gap runs straight through them. Their extent has to be
    # measured after a draw, since it depends on the rendered text size.
    if has_kl and kl_axes:
        fig.canvas.draw()
        rend = fig.canvas.get_renderer()
        inv = fig.transFigure.inverted()
        x_lab = min(t.get_window_extent(rend).transformed(inv).x0
                    for t in kl_labels)
        x_msp = top_axes[-1].get_position().x1
        y0 = min(a_.get_position().y0 for a_ in bot_axes + kl_axes)
        y1 = max(a_.get_position().y1 for a_ in top_axes + kl_axes)
        x_sep = max(0.5 * (x_msp + x_lab), x_msp + 0.004)
        fig.add_artist(plt.Line2D([x_sep] * 2, [y0, y1],
                                  transform=fig.transFigure, color="0.8",
                                  lw=0.7, zorder=0))

    # setup group titles, centred on the two panels they span (hardcoded x
    # positions drift as soon as the column count or widths change)
    if outside or nset != 2:
        xs_title = []
        for i, setup in enumerate(setups):
            b0 = top_axes[2 * i].get_position()
            b1 = top_axes[2 * i + 1].get_position()
            xs_title.append((setup, 0.5 * (b0.x0 + b1.x1)))
    else:
        xs_title = list(zip(setups, (0.29, 0.765)))
    for setup, xc in xs_title:
        fig.text(xc, 0.97, f"setup {SETUP_LETTERS[setup]}:  {results[setup]['col_title']}",
                 ha="center", va="top", fontsize=10.5, fontweight="bold")
    fig.savefig(out_path, dpi=200)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"wrote {out_path} (+.pdf)")


def _tagged_title(ax, tag, text, fontsize=7.5, y=1.0, borderpad=0.3, sep=3.5):
    """Bold tag + descriptive title above the axes, positioned so the
    DESCRIPTIVE half is what sits centred over the panel; the tag hangs off to
    its left, the way a panel letter normally does. Centring the pair as a
    group instead leaves the title itself right of the panel's midline by half
    the tag's width, which reads as off-centre.

    They are packed rather than written as one string because the tag is held
    at TAG_FS across the paper's figures while the descriptive half takes
    whatever its panel is wide enough for; one string would force one size on
    both."""
    tagbox = TextArea(tag, textprops=dict(fontsize=TAG_FS, fontweight="bold"))
    group = HPacker(children=[tagbox,
                              TextArea(text, textprops=dict(fontsize=fontsize))],
                    pad=0, sep=sep, align="baseline")
    # y is an axes-fraction anchor, so a panel whose axes box was shortened can
    # push its title back up onto the row's title line
    box = AnchoredOffsetbox(loc="lower center", child=group, pad=0,
                            borderpad=borderpad, frameon=False,
                            bbox_to_anchor=(0.5, y), bbox_transform=ax.transAxes)
    ax.add_artist(box)
    # then shift left by half of what the tag adds, so the midpoint of the
    # descriptive half lands on the axes' midpoint. Measured, not estimated:
    # the tag's width depends on the glyphs and the font size.
    rend = ax.figure.canvas.get_renderer()
    dx = 0.5 * (tagbox.get_window_extent(rend).width + sep) \
        / ax.get_window_extent(rend).width
    box.set_bbox_to_anchor((0.5 - dx, y), ax.transAxes)


def make_msp_row_figure(results, out_path: Path, setup, kl_setups=None,
                        bar_style=(0.35, 0.28)):
    """Single-ROW layout of one setup: each factor's Bayes-optimal MSP and the
    transformer's reconstruction sit side by side instead of stacking, so the
    whole figure is one row -- 2 panels per factor, a spacer between the factor
    pairs, then the KL panel. Half the height of make_msp_figure at the cost of
    smaller panels, which is the trade the main text wants."""
    r = results[setup]
    has_kl = kl_setups is not None
    #        T truth/recon   gap   U truth/recon    gap   KL
    widths = [1.0, 1.0, 0.15, 1.0, 1.0] + ([0.40, 1.50] if has_kl else [])
    ncol = len(widths)
    # the KL panel is the widest: two bar groups, each annotated, in a panel
    # only as tall as one MSP row. The 0.40 spacer before it carries its y tick
    # labels and ylabel.
    fig_w, fig_h = (6.6, 1.52) if has_kl else (4.9, 1.52)
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(1, ncol, wspace=0.08, width_ratios=widths,
                          left=0.028, right=0.995, top=0.783, bottom=0.112)
    tags = [f"({c})" for c in string.ascii_lowercase]
    # gs columns holding the four MSP panels (2 and 5 are spacers)
    msp_cols = [0, 1, 3, 4]
    msp_axes = []
    for i, (fi, kind) in enumerate([(0, "truth"), (0, "recon"),
                                    (1, "truth"), (1, "recon")]):
        fac = r["factors"][fi]
        final_xy = fac["per_step"][CKPT_FINAL][0]
        lims = _lims(np.vstack([fac["coords_theory"], final_xy]))
        ax = fig.add_subplot(gs[0, msp_cols[i]])
        msp_axes.append(ax)
        xy = fac["coords_theory"] if kind == "truth" else final_xy
        _scatter(ax, xy, fac["colors"], lims, s=1.1, barcode=fac["barcode"],
                 bar_lw=bar_style[0], bar_alpha=bar_style[1])
        # tag folded into a left-aligned title: at this panel width a centred
        # title fills the whole band, so a tag beside it either overhangs into
        # the neighbouring panel or lands on the title itself
        _tagged_title(ax, tags[i],
                      "Bayes-optimal" if kind == "truth" else "transformer")
        if kind == "truth":
            _color_legend(ax, fac["n_color"], fac["kind"], fac.get("col_dirs"))
        else:
            # 6pt, not the stacked layout's 7.5: at this panel width a
            # larger box reaches far enough right to clip the fractal2
            # factor's lowest streak
            ax.text(0.03, 0.02,
                    f"$R^2={fac['per_step'][CKPT_FINAL][1]:.3f}$",
                    transform=ax.transAxes, fontsize=6.0, ha="left",
                    va="bottom",
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="0.6",
                              lw=0.4, alpha=0.9))
    kl_ax = None
    if has_kl:
        kl_ax = fig.add_subplot(gs[0, ncol - 1])
        # a row-height panel has to stack the colour key, the gap-closed
        # annotations and the bars in the height of a single MSP panel: the key
        # goes across one centred line at the top, and the annotations are
        # pinned just under it on one line rather than placed by KL value
        kl_ymax = 10.0
        _kl_panel(kl_ax, kc.load(), kl_setups[setup], ymax=kl_ymax,
                  gap_axes_y=0.88, gap_oneline=True, key_y0=1.05,
                  key_row=True, label_every=2,
                  fs_lab=7, fs_tick=6, fs_gap=6.5, fs_ylab=7.5)
        # Draw the axis only as tall as the decades it shows: the bars stop
        # under 1 nat, so a cell-height log axis is mostly empty. Height scales
        # with the decade count, so nats-per-inch is unchanged -- and the axis
        # stays bottom-anchored, keeping its x labels on the row's baseline.
        cell = kl_ax.get_position()
        kl_ax.set_position([cell.x0, cell.y0, cell.width,
                            cell.height * math.log10(kl_ymax / KL_YMIN)
                            / KL_DECADES_FULL])
        # the title follows the axis top down, so anchor it back up onto the
        # row's title line
        pos = kl_ax.get_position()
        drop = (cell.y1 - pos.y1) / pos.height      # in axes fractions
        _tagged_title(kl_ax, tags[4], "Belief patching", y=1.0 + drop)

    # factor group titles, centred over each factor's pair of panels
    fig.canvas.draw()
    for j, fi in enumerate((0, 1)):
        fac = r["factors"][fi]
        b0 = msp_axes[2 * j].get_position()
        b1 = msp_axes[2 * j + 1].get_position()
        fig.text(0.5 * (b0.x0 + b1.x1), 0.995,
                 f"{r['row_labels'][fi]}, $d\\!=\\!{fac['dim'] - 1}$",
                 ha="center", va="top", fontsize=9, fontweight="bold")
    fig.savefig(out_path, dpi=200)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"wrote {out_path} (+.pdf)")


def make_timeline_figure(results, out_path: Path, setups=None,
                         name_setup=True):
    """Truth MSP + per-checkpoint reconstructions for both factors of each
    setup in `setups` (indices into results; default SETUP_ORDER, i.e. both).
    With a single setup the figure is two rows, sized so the panels keep the
    four-row version's square aspect, and names the setup in a title line
    instead of the rotated side label. name_setup=False drops that title
    altogether (the main text names the loop in the caption and reserves the
    setup letters for the appendix) and gives its band back to the panels."""
    setups = list(SETUP_ORDER if setups is None else setups)
    single = len(setups) == 1
    steps = [s for s in TIMELINE_STEPS
             if s in results[setups[0]]["factors"][0]["per_step"]]
    ncol = 1 + len(steps)
    nrow = 2 * len(setups)  # (T, U) per setup
    titled = not single or name_setup
    # the single-setup figure prints at 5.5/7.0 of its authored size, so its
    # fonts are authored ~1.4x larger to land at the appendix version's size
    k = 1.4 if single else 1.0
    top = 0.90 if (single and titled) else 0.93
    # the larger legend label under the bottom-left panel needs a bottom
    # margin; the extra height keeps the panels the same size
    bottom = 0.075 if single else 0.02
    # without the rotated "setup X" label the left margin only has to hold
    # the row labels
    left = 0.045 if single else 0.09
    fig = plt.figure(figsize=(7.0, (3.2 if titled else 3.1) if single else 5.9))
    gs = fig.add_gridspec(nrow, ncol, hspace=0.10, wspace=0.06,
                          left=left, right=0.995, top=top, bottom=bottom)
    r2_bbox = dict(boxstyle="round,pad=0.12", fc="white", ec="0.6", lw=0.35, alpha=0.9)
    sep_axes = {}  # (row) -> (col0 ax, col1 ax) for the truth/recon separator
    for j, setup in enumerate(setups):
        r = results[setup]
        for fi, fac in enumerate(r["factors"]):
            row = 2 * j + fi
            lims = _lims(fac["coords_theory"])
            ax = fig.add_subplot(gs[row, 0])
            _scatter(ax, fac["coords_theory"], fac["colors"], lims, s=0.7, alpha=0.8,
                     barcode=fac["barcode"])
            _color_legend(ax, fac["n_color"], fac["kind"], fac.get("col_dirs"),
                          fs=6.0 * k)  # legend on truth column
            if row == 0:
                ax.set_title("MSP (truth)", fontsize=7 * k)
            ax.text(-0.10, 0.5, r["row_labels"][fi], transform=ax.transAxes,
                    rotation=90, va="center", ha="right", fontsize=6.5 * k)
            if fi == 0 and not single:
                ax.text(-0.28, 0.0, f"setup {SETUP_LETTERS[setup]}",
                        transform=ax.transAxes, rotation=90, va="center",
                        ha="right", fontsize=7.5, fontweight="bold")
            sep_axes.setdefault(row, [ax, None])
            for ci, step in enumerate(steps):
                xy, r2 = fac["per_step"][step]
                ax = fig.add_subplot(gs[row, 1 + ci])
                if ci == 0:
                    sep_axes[row][1] = ax
                _scatter(ax, xy, fac["colors"], lims, s=0.7, alpha=0.8,
                         barcode=fac["barcode"])
                if row == 0:
                    lbl = f"step {step//1000}k" if step >= 1000 else f"step {step}"
                    ax.set_title(lbl, fontsize=7 * k)
                ax.text(0.03, 0.04, f"$R^2$={r2:.3f}", transform=ax.transAxes,
                        fontsize=5.2 * k, ha="left", va="bottom", bbox=r2_bbox)
    # gray separator: Bayes-optimal truth column | network reconstructions
    left_ax, right_ax = sep_axes[0]
    x = 0.5 * (left_ax.get_position().x1 + right_ax.get_position().x0)
    y0 = min(a[0].get_position().y0 for a in sep_axes.values())
    y1 = max(a[0].get_position().y1 for a in sep_axes.values()) + 0.035
    fig.add_artist(plt.Line2D([x, x], [y0, y1], transform=fig.transFigure,
                              color="0.55", lw=1.0))
    if single and titled:
        # centred on the panel block (the row labels take the left margin)
        x0 = min(a[0].get_position().x0 for a in sep_axes.values())
        x1 = fig.axes[-1].get_position().x1
        fig.text(0.5 * (x0 + x1), 0.995,
                 f"setup {SETUP_LETTERS[setups[0]]}:  {results[setups[0]]['col_title']}",
                 ha="center", va="top", fontsize=9.5, fontweight="bold")
    fig.savefig(out_path, dpi=200)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"wrote {out_path} (+.pdf)")


def make_steering_figure(out_path: Path, setup, kl_setups, name_setup=False):
    """Standalone belief-patching panel for one setup: the same joint-KL bars
    as the stacked layout's lone KL column, sized for a half-column wrapfigure
    (0.48 of ICLR's 5.5in text width)."""
    fig = plt.figure(figsize=(2.64, 2.35))
    ax = fig.add_axes([0.215, 0.125, 0.765, 0.78])
    _kl_panel(ax, kc.load(), kl_setups[setup], ymax=60.0,
              key_y0=0.99, key_dy=0.05)
    ax.set_title("Belief patching" + (f", setup {SETUP_LETTERS[setup]}"
                                      if name_setup else ""), fontsize=9)
    fig.savefig(out_path, dpi=200)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"wrote {out_path} (+.pdf)")



def make_figures(out_dir: Path, pair: str = "fractal2"):
    """Collect both setups of `pair` and write every figure into out_dir."""
    if pair not in PAIRS:
        raise SystemExit(f"pair must be one of {sorted(PAIRS)}, got {pair!r}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = PAIRS[pair]
    runs = [(letter, resolve_run(name)) for letter, name in spec["runs"]]
    stem = spec["stem"]

    missing = [rd for _, rd in runs if not (rd / "checkpoints").is_dir()]
    if missing:
        raise SystemExit("missing runs: " + ", ".join(str(m.name) for m in missing))

    print(f"Processing feedback runs (pair={pair}): "
          f"{', '.join(rd.name for _, rd in runs)}")
    results = [collect(rd, spec.get("orient", False)) for _, rd in runs]
    # the drawers read each setup's titles from its own results entry
    for i, r in enumerate(results):
        r["col_title"] = spec["col_titles"][i]
        r["row_labels"] = spec["row_labels"][i]
    bar_style = spec.get("bar_style", (0.35, 0.28))
    # both setups: the appendix companion (and, for pairs without a
    # `main_setup`, still the figure the paper shows)
    make_msp_figure(results, out_dir / f"{stem}.png", spec.get("kl_setups"),
                    bar_style, setups=SETUP_ORDER)
    main_setup = spec.get("main_setup")
    if main_setup is not None:
        make_msp_row_figure(
            results, out_dir / f"{stem}_setup{SETUP_LETTERS[main_setup]}.png",
            main_setup, spec.get("kl_setups"), bar_style)
        # main text: the timeline of the main setup alone, and the steering
        # bars as their own half-column figure; the other setup's timeline
        # goes to the appendix
        letter = SETUP_LETTERS[main_setup]
        make_timeline_figure(results, out_dir / f"{stem}_setup{letter}_timeline.png",
                             setups=[main_setup], name_setup=False)
        for other in range(len(results)):
            if other != main_setup:
                make_timeline_figure(
                    results,
                    out_dir / f"{stem}_setup{SETUP_LETTERS[other]}_timeline.png",
                    setups=[other])
        if spec.get("kl_setups"):
            make_steering_figure(out_dir / f"figure_steering_setup{letter}.png",
                                 main_setup, spec["kl_setups"])
    make_timeline_figure(results, out_dir / f"{stem}_timeline.png")
    return results
