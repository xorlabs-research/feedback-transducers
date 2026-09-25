"""Probe selectivity controls for the feedback-transducer feedback runs.

Reviewer objection this answers: "a held-out R^2 of 0.999 read off a 120-dim
residual stream is meaningless without selectivity controls -- a random
120-dim feature bank linearly fits any low-dimensional target."

For each log-spaced checkpoint of the two mess3<->fractal2 feedback runs
(the pair shown in the paper's Fig. fig:feedback_msp) we fit the SAME linear
probe used by the paper figures onto five different targets and plot the
held-out R^2 against the training step:

  true T          agent T's belief                 (factor 0, dim |R|)
  true U          action-delayed agent U's belief   (factor 1, dim |S|*|X|)
  joint (Kron)    the tensor-product joint belief   (dim |R|*|S|*|X|)
  shifted (t-2)   TRUE belief of the same sequence but 2 positions earlier
                  (temporal specificity control), mean over the two factors
  shuffled seqs   beliefs taken from a random OTHER sequence of the batch
                  (targets decoupled from the activations), mean over factors

Protocol (identical to plotting.feedback_msp.collect, which generates the
paper's MSP/timeline figures):
  * 1024 sequences sampled from the exact generative process with
    fwh_core...generate_data_batch_with_full_history, PRNGKey(12345);
    n_ctx = 9 (BOS + 8), BOS position dropped -> 7 token positions/sequence.
  * activations: blocks.{n_layers-1}.hook_resid_post (d_model = 120), the site
    used by the MSP and timeline figures.
  * probe: ordinary least squares with intercept (np.linalg.lstsq, rcond=None)
    -- the same estimator as fwh_core.analysis.linear_regression.linear_regression
    (fit_intercept=True). No shrinkage.
  * split: 80/20 by SEQUENCE (819/205 sequences = 5733/1435 token rows), so no
    token from a training sequence appears in the test set.
  * R^2 = 1 - SS_res/SS_tot pooled over all target dimensions on the held-out
    rows -- the fwh_core `_regression_metrics` convention.

Outputs (next to this script):
  figure_probe_selectivity.{pdf,png}, probe_selectivity_controls.json

Usage:
  python experiments/fig11_probe_controls/probe_selectivity_controls/probe_selectivity_controls.py
  ... --steps 0,10,100,1000,10000,50000   (subset, for a quick pass)
  ... --plot-only                          (re-draw from the saved JSON)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

from plotting.lib import feedback_runs as ff  # noqa: E402
from plotting.lib.feedback_runs import (  # noqa: E402
    BATCH, SEED, load_cfg_proc, load_model, sample_batch, stack_beliefs,
    acts_at_last_resid_post, fit_predict_linear, kron_per_token,
)

from plotting.lib.runs import run_path  # noqa: E402  (honors FACTORED_REPS_DATA)

SCHEMA_VERSION = 2  # v1 = single canonical seed per setup; v2 = 5 seeds per setup

# The paper's main feedback figure (Figures/figure_feedback_msp_fractal2.pdf)
# is built from these two setups. Each is run at five seeds: the canonical one
# (listed first, the run the figure itself uses) plus four reruns.
SETUPS = [
    ("A", "setup A:  $T$ = fractal2 (dim 2),  $U$ = mess3 (dim 9)",
     ["feedback_fractal2_mess3_canonical_50k"]
     + [f"feedback_fractal2_mess3_canonical_s{s}_50k" for s in range(1001, 1005)]),
    ("B", "setup B:  $T$ = mess3 (dim 3),  $U$ = fractal2 (dim 4)",
     ["feedback_mess3_fractal2_canonical_50k"]
     + [f"feedback_mess3_fractal2_canonical_s{s}_50k" for s in range(1301, 1305)]),
]

# canonical-seed view, kept so the old single-seed code paths still work

LAG = 2  # time-shifted control: belief at position t-LAG as target at t

TARGETS = [  # key, label, color, linestyle
    ("true_f0", "true belief $T$", "#4C72B0", "-"),
    ("true_f1", "true belief $U$", "#DD8452", "-"),
    ("joint", "joint $T\\otimes U$ (Kronecker)", "#55A868", "-"),
    ("shift", f"time-shifted ($t{{-}}{LAG}$)", "#8172B3", "--"),
    ("shuffle", "shuffled sequences", "#C44E52", ":"),
]


def _derangement(n: int, rng: np.random.Generator) -> np.ndarray:
    """Random permutation with no fixed point (no sequence keeps its own
    belief), so the shuffled control has zero true-pair leakage."""
    perm = rng.permutation(n)
    while True:
        fixed = np.flatnonzero(perm == np.arange(n))
        if fixed.size == 0:
            return perm
        if fixed.size == 1:
            j = int(rng.integers(n - 1))
            j = j + 1 if j >= fixed[0] else j
            perm[[fixed[0], j]] = perm[[j, fixed[0]]]
        else:
            perm[fixed] = perm[np.roll(fixed, 1)]


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pooled-over-dimensions R^2, the fwh_core `_regression_metrics` form."""
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean(0)) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def available_steps(run_dir: Path) -> list[int]:
    return sorted(int(p.name) for p in (run_dir / "checkpoints").iterdir()
                  if p.name.isdigit() and (p / "model.pt").exists())


def collect(run_dir: Path, steps: list[int] | None = None) -> dict:
    cfg, process = load_cfg_proc(run_dir)
    inputs, beliefs = sample_batch(cfg, process)
    b_list = stack_beliefs(beliefs)
    action_size = ff.u_action_size(cfg)

    # drop BOS -> (B, T, D)
    bf = [np.asarray(b[:, 1:]) for b in b_list]
    B, T = bf[0].shape[0], bf[0].shape[1]
    dims = [b.shape[-1] for b in bf]

    # --- targets -----------------------------------------------------------
    # true factored beliefs
    Y = {f"true_f{i}": b.reshape(B * T, -1) for i, b in enumerate(bf)}
    # joint Kronecker belief (dim prod(dims)); one free dim less than its
    # ambient size since it lives on the simplex, like the factors.
    Y["joint"] = kron_per_token(bf).reshape(B * T, -1)

    # shuffled-sequence control: each factor's belief comes from a different
    # sequence (same position). Independent derangement per factor.
    rng = np.random.default_rng(SEED + 7)
    for i, b in enumerate(bf):
        Y[f"shuffle_f{i}"] = b[_derangement(B, rng)].reshape(B * T, -1)

    # time-shifted control: target at row (b, t) is the belief at (b, t-LAG).
    # Only positions t >= LAG are valid; a position mask selects those rows.
    pos = np.tile(np.arange(T), B)
    valid = pos >= LAG
    for i, b in enumerate(bf):
        sh = np.empty_like(b)
        sh[:, LAG:] = b[:, :-LAG]
        sh[:, :LAG] = 0.0  # masked out below
        Y[f"shift_f{i}"] = sh.reshape(B * T, -1)

    # --- 80/20 split by sequence ------------------------------------------
    rng2 = np.random.default_rng(SEED + 1)  # same stream as figure_feedback_msp
    idx = np.arange(B)
    rng2.shuffle(idx)
    cut = int(0.8 * B)
    tr_mask = np.zeros(B, dtype=bool)
    tr_mask[idx[:cut]] = True
    is_tr = tr_mask[np.repeat(np.arange(B), T)]
    is_te = ~is_tr

    steps = steps if steps is not None else available_steps(run_dir)
    out = {
        "run": run_dir.name,
        "seed": int(cfg.get("seed", -1)),
        "n_seq": int(B), "n_pos": int(T),
        "dims": dims, "action_size": int(action_size),
        "d_model": int(cfg.predictive_model.instance.cfg.d_model),
        "n_layers": int(cfg.predictive_model.instance.cfg.n_layers),
        "joint_dim": int(Y["joint"].shape[1]),
        "n_train_rows": int(is_tr.sum()), "n_test_rows": int(is_te.sum()),
        "lag": LAG,
        "steps": [], "r2": {},
    }

    def _fit(key: str, mask: np.ndarray | None = None) -> float:
        y = Y[key]
        tr = is_tr if mask is None else (is_tr & mask)
        te = is_te if mask is None else (is_te & mask)
        pred = fit_predict_linear(A[tr], y[tr], A)
        return _r2(y[te], pred[te])

    for step in steps:
        ckpt = run_dir / f"checkpoints/{step}/model.pt"
        if not ckpt.exists():
            print(f"    missing ckpt {step}")
            continue
        t0 = time.time()
        model = load_model(cfg, ckpt)
        A = acts_at_last_resid_post(model, inputs)[:, 1:]
        A = A.reshape(B * T, -1)

        row = {}
        for i in range(len(bf)):
            row[f"true_f{i}"] = _fit(f"true_f{i}")
            row[f"shuffle_f{i}"] = _fit(f"shuffle_f{i}")
            row[f"shift_f{i}"] = _fit(f"shift_f{i}", valid)
            # true belief restricted to the shifted control's position subset,
            # so the shift curve has an exactly comparable reference.
            row[f"true_f{i}_lagpos"] = _fit(f"true_f{i}", valid)
        row["joint"] = _fit("joint")
        row["shuffle"] = float(np.mean([row[f"shuffle_f{i}"] for i in range(len(bf))]))
        row["shift"] = float(np.mean([row[f"shift_f{i}"] for i in range(len(bf))]))
        out["steps"].append(int(step))
        for k, v in row.items():
            out["r2"].setdefault(k, []).append(float(v))
        print(f"    step {step:>6}: " + "  ".join(
            f"{k}={row[k]:.4f}" for k in ("true_f0", "true_f1", "joint", "shift", "shuffle"))
            + f"   ({time.time() - t0:.1f}s)")
    return out


def _symlog_fmt(v, _pos):
    if v == 0:
        return "0"
    return r"$10^{%d}$" % int(round(np.log10(v)))


def _symlog_x(ax):
    ax.set_xscale("symlog", linthresh=10, linscale=0.45)
    ax.set_xlim(0, 60000)
    ax.xaxis.set_major_locator(FixedLocator([0, 1e1, 1e2, 1e3, 1e4]))
    ax.xaxis.set_major_formatter(FuncFormatter(_symlog_fmt))


def common_steps(seed_results: list[dict]) -> np.ndarray:
    """Steps present in every seed of a setup (they agree in practice)."""
    sets = [set(r["steps"]) for r in seed_results]
    return np.array(sorted(set.intersection(*sets)), dtype=int)


def seed_band(seed_results: list[dict], key: str, steps: np.ndarray):
    """(mean, min, max) of an R^2 curve across seeds, on the common grid."""
    rows = []
    for r in seed_results:
        lut = dict(zip(r["steps"], r["r2"][key]))
        rows.append([lut[int(s)] for s in steps])
    a = np.asarray(rows, dtype=float)
    return a.mean(0), a.min(0), a.max(0)


def make_figure(results: list[list[dict]], out_png: Path):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.35), sharey=True)
    for ax, seed_res, (_tag, title, _dirs) in zip(axes, results, SETUPS):
        steps = common_steps(seed_res).astype(float)
        # inset: the three near-saturating curves, magnified
        ins = ax.inset_axes([0.405, 0.135, 0.565, 0.375])
        for key, label, color, ls in TARGETS:
            y, lo, hi = seed_band(seed_res, key, steps.astype(int))
            ax.fill_between(steps, lo, hi, color=color, alpha=0.2, lw=0)
            ax.plot(steps, y, ls, lw=1.5, color=color, label=label,
                    marker="o", ms=1.5, mew=0)
            if key in ("true_f0", "true_f1", "joint"):
                ins.fill_between(steps, lo, hi, color=color, alpha=0.2, lw=0)
                ins.plot(steps, y, ls, lw=1.1, color=color, marker="o",
                         ms=1.2, mew=0)
        ax.axhline(0.0, color="0.5", lw=0.7, ls="-", zorder=0)
        _symlog_x(ax)
        ax.set_xlabel("optimizer step")
        ax.set_title(title, fontsize=8.5)
        ax.grid(alpha=0.3, which="both")
        ax.tick_params(which="both", labelsize=7)
        ax.set_ylim(-0.08, 1.045)

        _symlog_x(ins)
        ins.set_ylim(0.955, 1.0015)
        ins.set_yticks([0.96, 0.98, 1.0])
        ins.set_yticklabels(["0.96", "0.98", "1.00"])
        ins.xaxis.set_major_locator(FixedLocator([0, 1e2, 1e4]))
        ins.tick_params(which="both", labelsize=5.2, length=2, pad=1.2)
        ins.grid(alpha=0.25, which="major")
        ins.set_title("zoom: true and joint targets", fontsize=5.6, pad=1.8)
        for sp in ins.spines.values():
            sp.set_linewidth(0.6)
    axes[0].set_ylabel("held-out $R^2$ of linear probe")
    for ax, lab in zip(axes, ("(a)", "(b)")):
        ax.text(0.015, 0.975, lab, transform=ax.transAxes, fontsize=10,
                fontweight="bold", va="top")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=7.0,
               frameon=False, handlelength=2.0, columnspacing=1.6,
               handletextpad=0.5, borderaxespad=0.2)
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    fig.savefig(out_png, dpi=220)
    fig.savefig(out_png.with_suffix(".pdf"))
    plt.close(fig)
    print(f"wrote {out_png} (+.pdf)")


def main():
    argv = sys.argv[1:]
    json_path = HERE / "probe_selectivity_controls.json"
    steps = None
    plot_only = "--plot-only" in argv
    for a in argv:
        if a.startswith("--steps"):
            v = a.split("=", 1)[1] if "=" in a else argv[argv.index(a) + 1]
            steps = [int(s) for s in v.split(",")]

    if plot_only:
        payload = json.loads(json_path.read_text())
        results = [s["seeds"] for s in payload["setups"]]
    else:
        results = []
        for tag, _title, dirs in SETUPS:
            per_seed = []
            for k, name in enumerate(dirs):
                kind = "canonical" if k == 0 else f"rerun {k}"
                print(f"[{tag}] {name}  ({kind})")
                per_seed.append(collect(run_path(name), steps))
            results.append(per_seed)
        json_path.write_text(json.dumps({
            "schema_version": SCHEMA_VERSION,
            "lag": LAG, "batch": BATCH, "seed": SEED,
            # v1 field: the canonical seed of each setup, unchanged in meaning
            "runs": [r[0] for r in results],
            "setups": [
                {"tag": tag, "title": title,
                 "n_seeds": len(per_seed),
                 "seeds": per_seed}
                for (tag, title, _dirs), per_seed in zip(SETUPS, results)
            ],
        }, indent=1))
        print(f"wrote {json_path}")

    make_figure(results, HERE / "figure_probe_selectivity.png")


if __name__ == "__main__":
    main()
