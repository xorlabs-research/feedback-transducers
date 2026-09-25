"""Layer-wise emergence of factored belief representations over training.

For the two mess3<->fractal2 feedback-loop runs behind the paper's
fig:feedback_msp (Figures/figure_feedback_msp_fractal2.pdf):

    setup A  feedback_fractal2_mess3_canonical_50k   (T = fractal2, U = mess3)
    setup B  feedback_mess3_fractal2_canonical_50k   (T = mess3,    U = fractal2)

we fit a held-out linear belief probe at every residual-stream layer boundary,
for a log-spaced set of training checkpoints, and record R^2 per
(setup, checkpoint, site, factor).

Probe protocol matches the paper's probe code
(fwh_core.analysis.linear_regression.layer_linear_regression /
 linear_regression): plain least squares with intercept, no feature
standardization, R^2 aggregated over target dimensions
(1 - sum SS_res / sum SS_tot).  The split is the same 80/20 by-sequence split
used by plotting.feedback_msp, so the final-step
ln-free resid_post numbers here reproduce that figure's annotations.

Outputs (next to this file):
    layerwise_r2.json / layerwise_r2.npz
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

from plotting.lib import feedback_runs as ft  # noqa: E402
from plotting.lib.feedback_runs import (  # noqa: E402
    load_cfg_proc, load_model, sample_batch, stack_beliefs, SEED,
)

from plotting.lib.runs import run_path  # noqa: E402  (honors FACTORED_REPS_DATA)

SCHEMA_VERSION = 2  # v1 = single canonical seed per setup; v2 = 5 seeds per setup

# Five seeds per setup: the canonical run (first, the one the paper figure
# itself is built from) plus four reruns.
RUNS = [
    ("A", ("$T$ (fractal2)", "$U$ (mess3)"),
     ["feedback_fractal2_mess3_canonical_50k"]
     + [f"feedback_fractal2_mess3_canonical_s{s}_50k" for s in range(1001, 1005)]),
    ("B", ("$T$ (mess3)", "$U$ (fractal2)"),
     ["feedback_mess3_fractal2_canonical_50k"]
     + [f"feedback_mess3_fractal2_canonical_s{s}_50k" for s in range(1301, 1305)]),
]

# log-spaced, from the run's own checkpoint schedule (78 ckpts exist locally)
STEPS = [0, 1, 3, 10, 30, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]

# residual-stream layer boundaries of the 4-layer model, in depth order
SITES = [
    ("blocks.0.hook_resid_pre", "embed"),
    ("blocks.0.hook_resid_post", "resid 1"),
    ("blocks.1.hook_resid_post", "resid 2"),
    ("blocks.2.hook_resid_post", "resid 3"),
    ("blocks.3.hook_resid_post", "resid 4"),
    ("ln_final.hook_normalized", "ln final"),
]


def fit_r2(x_tr, y_tr, x_te, y_te) -> float:
    """Held-out R^2 of a least-squares linear probe with intercept.

    Same estimator and same aggregate-R^2 definition as
    fwh_core.analysis.linear_regression.linear_regression."""
    x1 = np.hstack([x_tr, np.ones((x_tr.shape[0], 1))])
    sol, *_ = np.linalg.lstsq(x1, y_tr, rcond=None)
    w, b = sol[:-1], sol[-1]
    pred = x_te @ w + b
    ss_res = float(np.sum((y_te - pred) ** 2))
    ss_tot = float(np.sum((y_te - y_te.mean(0)) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def acts_all_sites(model, inputs):
    with torch.no_grad():
        _, cache = model.run_with_cache(inputs, return_type="logits")
    out = {}
    for key, _ in SITES:
        a = cache[key].cpu().numpy()[:, 1:]      # drop BOS position
        out[key] = a.reshape(-1, a.shape[-1]).astype(np.float64)
    return out


def run_setup(run_dir: Path) -> dict:
    cfg, process = load_cfg_proc(run_dir)
    inputs, beliefs = sample_batch(cfg, process)
    b_list = stack_beliefs(beliefs)
    targets = [b[:, 1:].reshape(-1, b.shape[-1]).astype(np.float64) for b in b_list]
    targets.append(np.hstack(targets))          # concatenated (joint) target

    n_seq = b_list[0].shape[0]
    n_pos = b_list[0].shape[1] - 1
    rng = np.random.default_rng(SEED + 1)       # same split as figure_feedback_msp
    idx = np.arange(n_seq)
    rng.shuffle(idx)
    cut = int(0.8 * n_seq)
    tr_seq = np.zeros(n_seq, dtype=bool)
    tr_seq[idx[:cut]] = True
    is_tr = tr_seq[np.repeat(np.arange(n_seq), n_pos)]
    is_te = ~is_tr

    dims = [int(t.shape[-1]) for t in targets]
    print(f"  {run_dir.name}: n_rows={targets[0].shape[0]}, factor dims={dims[:-1]}")

    r2 = np.full((len(STEPS), len(SITES), len(targets)), np.nan)
    for si, step in enumerate(STEPS):
        ckpt = run_dir / f"checkpoints/{step}/model.pt"
        if not ckpt.exists():
            print(f"    missing ckpt {step}")
            continue
        model = load_model(cfg, ckpt)
        acts = acts_all_sites(model, inputs)
        for li, (key, _) in enumerate(SITES):
            x = acts[key]
            for fi, y in enumerate(targets):
                r2[si, li, fi] = fit_r2(x[is_tr], y[is_tr], x[is_te], y[is_te])
        print(f"    step {step:>6}: "
              + "  ".join(f"{lab}=({r2[si, li, 0]:.3f},{r2[si, li, 1]:.3f})"
                          for li, (_, lab) in enumerate(SITES)))
    return dict(r2=r2, dims=dims, n_rows=int(targets[0].shape[0]),
                seed=int(cfg.get("seed", -1)))


def main():
    out = {}
    for name, labels, run_names in RUNS:
        per_seed = []
        for k, rn in enumerate(run_names):
            run_dir = run_path(rn)
            if not (run_dir / "checkpoints").is_dir():
                raise SystemExit(f"missing checkpoints for {run_dir}")
            print(f"setup {name} ({'canonical' if k == 0 else f'rerun {k}'}): {rn}")
            res = run_setup(run_dir)
            res["run"] = rn
            per_seed.append(res)
        out[name] = {
            "factor_labels": list(labels),
            "runs": [r["run"] for r in per_seed],
            "seeds": [r["seed"] for r in per_seed],
            "dims": per_seed[0]["dims"],
            "n_rows": per_seed[0]["n_rows"],
            # [seed, step, site, factor]
            "r2_seeds": np.stack([r["r2"] for r in per_seed]),
        }

    payload = {
        "schema_version": SCHEMA_VERSION,
        "steps": STEPS,
        "sites": [k for k, _ in SITES],
        "site_labels": [l for _, l in SITES],
        "factors": ["T", "U", "concat"],
        "setups": {
            k: {"run": v["runs"][0],          # v1 field: canonical run
                "runs": v["runs"], "seeds": v["seeds"],
                "n_seeds": len(v["runs"]),
                "factor_labels": v["factor_labels"],
                "dims": v["dims"], "n_rows": v["n_rows"],
                "r2": v["r2_seeds"][0].tolist(),        # v1 field: canonical seed
                "r2_seeds": v["r2_seeds"].tolist()}     # [seed, step, site, factor]
            for k, v in out.items()
        },
        "protocol": {
            "n_sequences": ft.BATCH, "seed": SEED,
            "split": "80/20 by sequence, BOS position excluded",
            "probe": "least squares with intercept; R^2 aggregated over target dims",
        },
    }
    (HERE / "layerwise_r2.json").write_text(json.dumps(payload, indent=1))
    np.savez(HERE / "layerwise_r2.npz",
             steps=np.array(STEPS),
             sites=np.array([k for k, _ in SITES]),
             site_labels=np.array([l for _, l in SITES]),
             **{f"r2_{k}": v["r2_seeds"][0] for k, v in out.items()},
             **{f"r2_seeds_{k}": v["r2_seeds"] for k, v in out.items()})
    print(f"wrote {HERE/'layerwise_r2.json'} and .npz")


if __name__ == "__main__":
    main()
