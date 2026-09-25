"""Joint-output KL of the paper's patching runs against two random controls.

The steering matrices reduce each intervention to a discrete label. This is the
continuous view, on the JOINT next-token distribution, with the references

    H_orig = Bayes( b_1 , ... , b_N )           all beliefs from this sequence
    H_mix  = Bayes( b_1 , .. b'_f .. , b_N )    factor f swapped in from a
                                                random donor sequence -- the
                                                patching target itself
    H_rand = Bayes( b'_1 , ... , b'_N )         every belief from the donor,
                                                i.e. a wholly unrelated sequence

and four KLs per edited factor (nats, means over evaluable interventions):

  baseline      KL(p_clean   || H_orig)  the model's fidelity when untouched --
                                         the floor any edit inherits
  measured      KL(p_patched || H_mix )  did the edit produce the mixed target?
  control-null  KL(p_clean   || H_mix )  what a write with NO effect would give
  control-rand  KL(p_clean   || H_rand)  scale: distance to an unrelated sequence

Reading: measured ~ baseline => the edit landed; measured ~ control-null => the
write did nothing; control-rand sets the dynamic range of the whole test.

Means rather than medians: the per-intervention KL is bimodal (landed / did not
land), so a median jumps between modes at rates near 0.5.

The donor pairing is drawn per run from mr.donor_rng(run), the same
order-independent generator the marginal read-out uses. For the mess3 loop that
is exactly the pairing behind the paper's steering matrix; the telephone
matrices were computed with a shared RNG whose draw depends on call order, so
their donor differs -- all four KLs here still use one pairing per run, which is
what makes them comparable to each other.

Writes steering_kl_controls_paper.json.

Run from repo root:
    python analysis/steering_kl_controls_paper.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from plotting.lib import feedback_runs as ft       # noqa: E402
import intervention_factorization as iv           # noqa: E402
import intervention_marginal_readout as mr        # noqa: E402
import intervention_telephone as it               # noqa: E402

TV_FLOOR = iv.TV_FLOOR

# The three setups the paper's steering figure reports, in panel order.
SETUPS = [
    # s904's mlflow record ends at step 35000 (see SEED_MANIFEST.md), but all
    # 78 checkpoints incl. step 50000 are present; this analysis needs only
    # the config plus the final checkpoint.
    ("loop_mess3", "loop, mess3$\\,\\leftrightarrow\\,$mess3",
     ["feedback_mess3_mess3_canonical_50k"]
     + [f"feedback_mess3_mess3_canonical_s{s}_50k" for s in (901, 902, 903, 904)]),
    ("telephone_mess3", "telephone, $3\\times$ mess3",
     it.FAMILIES["telephone"]),
    ("telephone_fractal2", "telephone, $5\\times$ fractal2",
     it.FAMILIES["fractal2_telephone"]),
    # The two sns loops, whose steering matrices are the ones that deviate.
    # Their joint KLs are the diagnosis: a discrete orig/cf label is only
    # meaningful when the model's own baseline fidelity is small relative to
    # the separation between the two references.
    ("loop_sns_mess3", "loop, sns$\\,\\to\\,$mess3",
     iv.SETUPS["B (T=sns, U=mess3)"]),
    ("loop_mess3_sns", "loop, mess3$\\,\\to\\,$sns",
     iv.SETUPS["A (T=mess3, U=sns)"]),
    # The mess3 <-> fractal2 loops: the same role-swapped pair with a factor
    # family that steers, which is what the MSP figure's KL column reports.
    ("loop_fractal2_mess3", "loop, fractal2$\\,\\to\\,$mess3",
     ["feedback_fractal2_mess3_canonical_50k"]
     + [f"feedback_fractal2_mess3_canonical_s{s}_50k" for s in (1001, 1002, 1003, 1004)]),
    ("loop_mess3_fractal2", "loop, mess3$\\,\\to\\,$fractal2",
     ["feedback_mess3_fractal2_canonical_50k"]
     + [f"feedback_mess3_fractal2_canonical_s{s}_50k" for s in (1301, 1302, 1303, 1304)]),
]


def fit_maps(resid, b_list, L):
    """Least-squares belief decoders E_f (belief -> residual) and read probes
    W_f (residual -> belief), jointly over all factors, as the steering
    interventions themselves use."""
    pos = np.arange(1, L)
    Xb = [b[:, pos].reshape(-1, b.shape[-1]) for b in b_list]
    X = np.concatenate(Xb, -1)
    Y = resid[:, pos].reshape(-1, resid.shape[-1])
    C, *_ = np.linalg.lstsq(np.concatenate([X, np.ones((len(X), 1))], -1), Y,
                            rcond=None)
    dims = [b.shape[-1] for b in b_list]
    offs = np.concatenate([[0], np.cumsum(dims)])
    E = {i: C[offs[i]:offs[i + 1]] for i in range(len(b_list))}
    Y1 = np.concatenate([Y, np.ones((len(Y), 1))], -1)
    W = {}
    for i in range(len(b_list)):
        Wf, *_ = np.linalg.lstsq(Y1, X[:, offs[i]:offs[i + 1]], rcond=None)
        W[i] = Wf[:-1]
    return E, W


def prepare(run_name: str) -> dict:
    """Everything an intervention on `run_name` needs: the model, the batch, the
    fitted maps, the donor pairing and the write itself.

    Shared with steering_orth_variants.py and the dose-response pipeline so
    there is exactly ONE definition of the splice.
    """
    run_dir = iv.run_path(run_name)
    cfg, process = ft.load_cfg_proc(run_dir)
    inputs, beliefs = ft.sample_batch(cfg, process)
    b_list = [np.asarray(b, dtype=np.float64) for b in ft.stack_beliefs(beliefs)]
    model = ft.load_model(cfg, mr.final_ckpt(run_dir))
    base = int(cfg.generative_process.base_vocab_size)
    n_layers = int(cfg.predictive_model.instance.cfg.n_layers)
    hook = f"blocks.{n_layers - 1}.hook_resid_post"
    B, L = inputs.shape
    tstar = L - 2

    with torch.no_grad():
        _, cache = model.run_with_cache(inputs, return_type="logits")
    resid = cache[hook].double().cpu().numpy()

    # Factors that can be edited: everything with a non-trivial belief. The
    # telephone's router factor is a 1-dimensional constant and is skipped
    # (its belief cannot be swapped), matching intervention_telephone.
    editable = [i for i, b in enumerate(b_list) if b.shape[-1] > 1]
    E, W = fit_maps(resid, b_list, L)

    def write_delta(i, db):
        """Belief-space edit written along E_i, UNPROJECTED -- what the paper's
        main-text belief-patching panels report. The appendix
        compares this against the encoder-orthogonalized variant
        (steering_orth_variants.py): neither is uniformly conservative -- the
        projection also removes intended-write mass where encoder spans
        overlap, while the raw write can corrupt partner encodings."""
        return db @ E[i]

    bt = [b[:, tstar] for b in b_list]
    perm = mr.donor_rng(run_name).permutation(B)
    bd = [b[perm] for b in bt]
    return dict(run=run_name, cfg=cfg, process=process, model=model,
                inputs=inputs, b_list=b_list, resid=resid, hook=hook,
                base=base, tstar=tstar, B=B, L=L, editable=editable,
                E=E, W=W, write_delta=write_delta, bt=bt, bd=bd)


def analyze(run_name: str) -> dict:
    """Four joint-output KLs per edited factor, for one run."""
    P = prepare(run_name)
    process, model, inputs = P["process"], P["model"], P["inputs"]
    base, tstar, hook = P["base"], P["tstar"], P["hook"]
    b_list, editable = P["b_list"], P["editable"]
    write_delta = P["write_delta"]
    bt, bd = P["bt"], P["bd"]

    p_clean = iv.model_dist(model, inputs, tstar, base)
    H_orig = iv.bayes_dist(process, tuple(bt))
    H_rand = iv.bayes_dist(process, tuple(bd))

    out = {"run": run_name, "vocab": base, "n_factors": len(b_list),
           "editable": editable, "dims": [b.shape[-1] for b in b_list],
           "edits": []}
    for i in editable:
        state = list(bt)
        state[i] = bd[i]
        H_mix = iv.bayes_dist(process, tuple(state))
        p_patched = iv.model_dist(model, inputs, tstar, base,
                                  edit=(hook, write_delta(i, bd[i] - bt[i])))
        # identifiability: the mixed target must be distinguishable from the
        # original at all, else there is nothing for the edit to hit
        m = 0.5 * np.abs(H_orig - H_mix).sum(-1) > TV_FLOOR

        def mean_kl(p, q):
            return float(np.mean(iv.kl(p, q)[m]))

        null = mean_kl(p_clean, H_mix)
        rec = {"factor": i, "n_eval": int(m.sum()),
               "baseline": mean_kl(p_clean, H_orig),
               "measured": mean_kl(p_patched, H_mix),
               "control_null": null,
               "control_rand": mean_kl(p_clean, H_rand)}
        rec["closed_gap"] = 1.0 - rec["measured"] / max(null, 1e-12)
        out["edits"].append(rec)
        print(f"     factor {i}: baseline {rec['baseline']:.4f}  "
              f"measured {rec['measured']:.4f}  null {null:.4f}  "
              f"rand {rec['control_rand']:.4f}  "
              f"gap closed {rec['closed_gap']:+.3f}  (n={rec['n_eval']})",
              flush=True)
    return out


def main() -> None:
    res = []
    for key, label, runs in SETUPS:
        print(f"== {key}: {len(runs)} run(s)", flush=True)
        for run in runs:
            print(f"   {run}", flush=True)
            r = analyze(run)
            r["setup"], r["label"] = key, label
            res.append(r)
    (HERE / "steering_kl_controls_paper.json").write_text(json.dumps(res, indent=2))
    print(f"\nwrote steering_kl_controls_paper.json ({len(res)} runs)")

    print("\n=== per setup (mean over seeds and edited factors, nats) ===")
    for key, label, _ in SETUPS:
        rows = [e for r in res if r["setup"] == key for e in r["edits"]]
        if not rows:
            continue
        g = {k: float(np.mean([e[k] for e in rows]))
             for k in ("baseline", "measured", "control_null", "control_rand")}
        print(f"  {key:20s} baseline {g['baseline']:.4f}  "
              f"measured {g['measured']:.4f}  null {g['control_null']:.4f}  "
              f"rand {g['control_rand']:.4f}  "
              f"gap closed {1 - g['measured'] / g['control_null']:+.3f}  "
              f"({len(rows)} interventions)")


if __name__ == "__main__":
    main()
