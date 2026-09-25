"""Figure: dose-response of belief steering along the interpolation path.

2x2 grid -- rows = the two mess3<->fractal2 feedback setups, columns = which
factor's belief is patched. x = lambda (edit dose), y = KL in nats (log).

  on-path (blue)      KL(p_edited(lambda) || Bayes at the INTERPOLATED tuple)
  to original (gold)  KL(p_edited(lambda) || Bayes at lambda = 0)
  to full cf (red)    KL(p_edited(lambda) || Bayes at lambda = 1)
  leakage (green)     KL of the UNTOUCHED factor's component vs its original
  floor (grey dashed) KL(p_clean || Bayes at lambda = 0), the model's own
                      unedited fidelity

Median over pooled interventions (5 seeds x ~700-970 evaluable interventions
each), IQR as a band. Colour conventions and font sizes follow
plotting/steering_kl_controls.py.

Reads dose_response.npz / .json next to this script; writes
figure_dose_response.{png,pdf} and dose_response_summary.json to --out-dir
(default: figures/ at the repo root).

Run from repo root:
    python experiments/fig17_dose_response/figure_dose_response.py [--out-dir DIR]
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent

FIG_W = 5.5
FS = 7.5          # panel title
FS_LAB = 8.0
FS_TICK = 6.0
FS_LEG = 6.0

CURVES = [
    ("on_path", "edited $\\|$ Bayes at $b_f(\\lambda)$  (on-path)",
     "#1f6f8b", "-", 1.6, 3),
    ("to_orig", "edited $\\|$ Bayes at $\\lambda{=}0$", "#b8860b", "--", 1.1, 2),
    ("to_full", "edited $\\|$ Bayes at $\\lambda{=}1$", "#c0392b", "--", 1.1, 2),
    ("leak", "untouched factor's component $\\|$ its original",
     "#2e8b57", ":", 1.1, 2),
]
PANEL_FACTOR = {"loop_fractal2_mess3": {0: "patch $b_T$ (fractal2)",
                                        1: "patch $b_U$ (mess3)"},
                "loop_mess3_fractal2": {0: "patch $b_T$ (mess3)",
                                        1: "patch $b_U$ (fractal2)"}}
# Setup letters follow the paper's convention: A = the mess3-env arrangement
# (T = mess3, U = fractal2), shown first; B = the role-swapped loop.
SETUP_TITLE = {"loop_mess3_fractal2": "A:  $T{=}$mess3, $U{=}$fractal2",
               "loop_fractal2_mess3": "B:  $T{=}$fractal2, $U{=}$mess3"}
ROW_ORDER = ["loop_mess3_fractal2", "loop_fractal2_mess3"]


def pooled(meta, npz, setup, factor):
    """Concatenate every seed's per-intervention values for one (setup, factor).

    Returns {quantity: [n_lambda, n_pooled]} plus the per-seed run names.
    """
    runs = [r["run"] for r in meta["runs"] if r["setup"] == setup]
    out = {}
    for q in ("on_path", "to_orig", "to_full", "null", "leak", "own",
              "floor"):
        out[q] = np.concatenate([npz[f"{r}|{factor}|{q}"] for r in runs], axis=1)
    return out, runs


def main(out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = json.loads((HERE / "dose_response.json").read_text())
    npz = np.load(HERE / "dose_response.npz")
    lam = np.asarray(meta["lambdas"])
    setups = [k for k in ROW_ORDER if k in {s["key"] for s in meta["setups"]}]

    fig, axes = plt.subplots(2, 2, figsize=(FIG_W, 4.35), sharex=True,
                             sharey=True)
    summary = {"lambdas": meta["lambdas"], "write": meta["write"],
               "tv_floor": meta["tv_floor"], "panels": []}

    for row, setup in enumerate(setups):
        for col, factor in enumerate((0, 1)):
            ax = axes[row, col]
            P, runs = pooled(meta, npz, setup, factor)
            n = P["on_path"].shape[1]
            floor = float(np.median(P["floor"]))

            for q, _, colour, ls, lw, z in CURVES:
                med = np.median(P[q], 1)
                lo = np.quantile(P[q], 0.25, 1)
                hi = np.quantile(P[q], 0.75, 1)
                ax.fill_between(lam, lo, hi, color=colour, alpha=0.16,
                                lw=0, zorder=z)
                ax.plot(lam, med, ls, color=colour, lw=lw, zorder=z + 3)

            ax.axhline(floor, color="0.45", ls=(0, (4, 2)), lw=1.0, zorder=1)
            ax.set_yscale("log")
            ax.set_ylim(8e-6, 3.0)
            ax.set_xlim(-0.03, 1.03)
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.tick_params(labelsize=FS_TICK)
            ax.grid(alpha=0.3, which="major", zorder=0)
            ax.set_axisbelow(True)
            ax.set_title(f"{PANEL_FACTOR[setup][factor]}\n"
                         f"$n$ = {n} interventions (5 seeds)",
                         fontsize=FS, pad=3)
            if row == 1:
                ax.set_xlabel("edit dose  $\\lambda$", fontsize=FS_LAB)
            if col == 0:
                ax.set_ylabel("KL  [nats]", fontsize=FS_LAB)

            def at(q, l):
                return float(np.median(P[q][list(lam).index(l)]))

            summary["panels"].append({
                "setup": setup, "factor": factor, "runs": runs,
                "n_interventions": n, "floor_median": floor,
                "median": {q: [float(x) for x in np.median(P[q], 1)]
                           for q in ("on_path", "to_orig", "to_full", "null",
                                     "leak", "own")},
                # the paper's headline statistic, per dose: ratio of MEANS
                # (the per-intervention KL is bimodal, so means are the stable
                # summary -- same convention as steering_kl_controls_paper).
                # Undefined at lambda = 0, where the null IS the measurement.
                "gap_closed": [float(1.0 - a / b) for a, b in
                               zip(np.mean(P["on_path"], 1),
                                   np.mean(P["null"], 1))],
                "q25": {q: [float(x) for x in np.quantile(P[q], 0.25, 1)]
                        for q in ("on_path", "to_orig", "to_full", "leak")},
                "q75": {q: [float(x) for x in np.quantile(P[q], 0.75, 1)]
                        for q in ("on_path", "to_orig", "to_full", "leak")},
                "mean": {q: [float(x) for x in np.mean(P[q], 1)]
                         for q in ("on_path", "to_orig", "to_full", "leak")},
                "on_path_over_floor": [float(x / floor)
                                       for x in np.median(P["on_path"], 1)],
                "headline": {
                    "on_path_at_half": at("on_path", 0.5),
                    "on_path_at_one": at("on_path", 1.0),
                    "to_orig_at_half": at("to_orig", 0.5),
                    "to_full_at_half": at("to_full", 0.5),
                    "ratio_half_to_floor": at("on_path", 0.5) / floor,
                    "ratio_one_to_floor": at("on_path", 1.0) / floor,
                    "refs_over_on_path_at_half":
                        min(at("to_orig", 0.5), at("to_full", 0.5))
                        / at("on_path", 0.5),
                },
            })

    handles = [Line2D([], [], color=c, ls=ls, lw=lw, label=lab)
               for _, lab, c, ls, lw, _ in CURVES]
    handles.append(Line2D([], [], color="0.45", ls=(0, (4, 2)), lw=1.0,
                          label="unedited model's fidelity floor"))
    fig.legend(handles=handles, fontsize=FS_LEG, loc="lower center", ncol=2,
               framealpha=0.0, handlelength=2.0, labelspacing=0.3,
               columnspacing=1.2, borderpad=0.2, handletextpad=0.6,
               bbox_to_anchor=(0.5, -0.005))
    fig.subplots_adjust(left=0.155, right=0.985, top=0.915, bottom=0.215,
                        hspace=0.42, wspace=0.09)
    # setup identity as a row label, so the two column titles cannot collide
    for row, setup in enumerate(setups):
        pos = axes[row, 0].get_position()
        fig.text(0.016, (pos.y0 + pos.y1) / 2, SETUP_TITLE[setup],
                 rotation=90, va="center", ha="left", fontsize=FS,
                 fontweight="bold", color="0.2")
    out = out_dir / "figure_dose_response.png"
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    (out_dir / "dose_response_summary.json").write_text(json.dumps(summary,
                                                                   indent=2))
    print(f"wrote {out} (+.pdf) and dose_response_summary.json")

    print("\n=== headline (pooled medians, nats) ===")
    for p in summary["panels"]:
        h = p["headline"]
        print(f"  {p['setup']:20s} f{p['factor']}  n={p['n_interventions']:4d}  "
              f"floor {p['floor_median']:.2e}  "
              f"on-path(0.5) {h['on_path_at_half']:.2e} "
              f"({h['ratio_half_to_floor']:5.1f}x floor)  "
              f"on-path(1.0) {h['on_path_at_one']:.2e} "
              f"({h['ratio_one_to_floor']:5.1f}x)  "
              f"nearest ref at 0.5 is {h['refs_over_on_path_at_half']:5.1f}x "
              f"the on-path KL")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="dose-response figure")
    ap.add_argument("--out-dir", default=str(HERE.parents[1] / "figures"))
    main(ap.parse_args().out_dir)
