"""Effective-rank inversion in TRAINED models (coarse-graining demonstrator).

Compares the final-layer residual-stream effective rank of three transformers
trained on the SAME underlying feedback loop (apx:coarse_graining) but with
different observation interfaces:

  cg_partition_full : joint token (x, y), vocab 9  -> beliefs factor,
                      exact factored cloud dims 7/8 @ CEV 80/95%
  cg_partition_x    : only x, vocab 3 (agent U hidden) -> entangled, 16/22
  cg_partition_y    : only y, vocab 3 (agent T hidden) -> entangled, 17/22

Theorem prediction: the coarse models, given strictly LESS information, need a
HIGHER-dimensional belief representation than the full-observation model.

Also reports (a) convergence vs each run's own Bayes floor, so the comparison
is between models that actually solved their tasks, and (b) the exact belief
cloud dimension for the same batch, as the reference the model should match.

Run from repo root:
    python analysis/coarse_graining_trained.py
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from plotting.lib import feedback_runs as ft  # noqa: E402
from plotting.lib.runs import run_path  # noqa: E402
# five replicates: the original permutation draw plus four independent
# draws with unique training seeds (see experiments/SEED_MANIFEST.md)
REPLICATES = [
    ("r0", {"full": "cg_partition_full_canonical_50k",
            "x": "cg_partition_x_canonical_50k",
            "y": "cg_partition_y_canonical_50k"}),
] + [(f"r{i}", {m: f"cg_r{i}_{m}_canonical_50k" for m in ("full", "x", "y")})
     for i in (1, 2, 3, 4)]
MODES = [("full", "full (x,y): vocab 9"),
         ("x", "x only (agent U hidden): vocab 3"),
         ("y", "y only (agent T hidden): vocab 3")]
RUNS = [(lab, runs[m]) for m, lab in MODES for _, runs in REPLICATES[:1]]
THRESHOLDS = (0.80, 0.95, 0.99)


def eff_dims(X, ps=THRESHOLDS):
    Xc = X - X.mean(0, keepdims=True)
    s = np.linalg.svd(Xc, compute_uv=False)
    cev = np.cumsum(s ** 2) / max((s ** 2).sum(), 1e-300)
    return {p: int(np.searchsorted(cev, p) + 1) for p in ps}, cev


def convergence(run_dir: Path):
    f = glob.glob(str(run_dir / "mlruns/*/*/metrics/eval/loss/step"))
    p = glob.glob(str(run_dir / "mlruns/*/*/params/optimal_loss/average"))
    if not f or not p:
        return None
    last = open(f[0]).read().splitlines()[-1].split()
    floor = float(open(p[0]).read())
    return {"final_loss": float(last[1]), "floor": floor,
            "gap": float(last[1]) - floor, "step": int(last[2])}


def analyze(run_name: str):
    run_dir = run_path(run_name)
    cfg, process = ft.load_cfg_proc(run_dir)
    inputs, beliefs = ft.sample_batch(cfg, process)
    model = ft.load_model(cfg, ft.final_ckpt(run_dir))
    n_layers = int(cfg.predictive_model.instance.cfg.n_layers)
    hook = f"blocks.{n_layers - 1}.hook_resid_post"
    with torch.no_grad():
        _, cache = model.run_with_cache(inputs, return_type="logits")
    acts = cache[hook].double().cpu().numpy()[:, 1:]      # skip BOS
    d_model, = acts.shape[-1:]
    model_dims, cev = eff_dims(acts.reshape(-1, d_model))

    # exact belief cloud on the same batch (ground truth the model must carry)
    b_list = [np.asarray(b, dtype=np.float64) for b in ft.stack_beliefs(beliefs)]
    if len(b_list) > 1:      # factored (full obs): concatenated per-factor rep
        Bel = np.concatenate([b[:, 1:] for b in b_list], -1)
        rep = f"factored concat ({'+'.join(str(b.shape[-1]) for b in b_list)})"
    else:                    # coarse: single entangled joint belief
        Bel = b_list[0][:, 1:]
        rep = f"joint belief ({b_list[0].shape[-1]}-simplex)"
    belief_dims, bcev = eff_dims(Bel.reshape(-1, Bel.shape[-1]))
    return {"run": run_name, "vocab": int(process.vocab_size),
            "representation": rep,
            "model_dims": {str(k): v for k, v in model_dims.items()},
            "belief_dims": {str(k): v for k, v in belief_dims.items()},
            "convergence": convergence(run_dir),
            "cev_model": [float(x) for x in cev[:60]],
            "cev_belief": [float(x) for x in bcev[:60]]}


def main():
    out = []
    for rep, runs in REPLICATES:
        for mode, label in MODES:
            r = analyze(runs[mode])
            r.update(label=label, mode=mode, replicate=rep)
            out.append(r)
            md, c = r["model_dims"], r["convergence"]
            print(f"  {rep} {mode:4s}: model {md['0.8']:3d}/{md['0.95']:3d}/{md['0.99']:3d}"
                  f"  belief {r['belief_dims']['0.8']:3d}/{r['belief_dims']['0.95']:3d}"
                  f"/{r['belief_dims']['0.99']:3d}  gap={c['gap']:+.1e}", flush=True)
    (HERE / "coarse_graining_trained.json").write_text(json.dumps(out, indent=2))

    print(f"\n=== across {len(REPLICATES)} replicates: mean [min-max] ===")
    hdr = f"{'mode':6s} " + " ".join(f"{'model @'+t:>18s}" for t in ("80%", "95%", "99%"))
    print(hdr)
    for mode, _ in MODES:
        rows = [r for r in out if r["mode"] == mode]
        cells = []
        for t in ("0.8", "0.95", "0.99"):
            v = np.array([r["model_dims"][t] for r in rows])
            cells.append(f"{v.mean():6.1f} [{v.min():2d}-{v.max():2d}]")
        bel = np.array([r["belief_dims"]["0.95"] for r in rows])
        gaps = np.array([r["convergence"]["gap"] for r in rows])
        print(f"{mode:6s} " + " ".join(f"{c:>18s}" for c in cells)
              + f"   belief@95={bel.mean():.1f} [{bel.min()}-{bel.max()}]"
              + f"   gap<={gaps.max():+.1e}")
    print(f"\nwrote {HERE / 'coarse_graining_trained.json'}")


if __name__ == "__main__":
    main()
