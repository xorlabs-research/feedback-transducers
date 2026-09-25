"""Does it matter which subspace the steering write is orthogonalized against?

The paper's splice writes db along factor f's belief ENCODER E_f (belief ->
residual) and then removes the component that the OTHER factors' read PROBES
W_g (residual -> belief) would pick up:

    Delta_probe = Delta - proj_{span(cols W_g, g != f)} Delta

But the probe directions and the encoder directions are different subspaces, and
it is the encoder rows that the network's activations actually carry factor g's
belief along -- so the causally relevant thing to protect is arguably
span(rows E_g), not span(cols W_g):

    Delta_enc = Delta - proj_{span(rows E_g, g != f)} Delta

This script measures all three schemes -- none / probe / encoder -- on the same
batch, maps and donor pairing, and reports for each:

  removed        fraction of ||Delta|| the orthogonalization takes away
  lambda         <Delta W_f, db> / ||db||^2: does the write still install
                 factor f's own belief after the projection?
  leak_probe     ||Delta W_g|| / ||db||          (zero by construction for probe)
  leak_enc       ||proj_{rows E_g} Delta|| / ||Delta||  (zero for encoder)
  gap closed     1 - KL(patched||H_mix) / KL(clean||H_mix) on the JOINT output,
                 i.e. the same quantity the belief-patching panels report

plus, per run, the subspace overlap between each factor's probe span and its own
encoder span -- the premise being tested: if those coincide, the two schemes
must agree.

Writes steering_orth_variants.json.

Run from repo root:
    python analysis/steering_orth_variants.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import intervention_factorization as iv           # noqa: E402
import intervention_telephone as it               # noqa: E402
import steering_kl_controls_paper as skc          # noqa: E402  (analysis/)

TV_FLOOR = skc.TV_FLOOR
SCHEMES = ("none", "probe", "encoder")

# the two-factor loops (where "the other factor" is a single partner) plus one
# telephone ring per family, so the N>2 case is represented too
GROUPS = [
    ("loop_mess3", ["feedback_mess3_mess3_canonical_50k"]
     + [f"feedback_mess3_mess3_canonical_s{s}_50k" for s in (901, 902, 903, 904)]),
    ("loop_fractal2_mess3", ["feedback_fractal2_mess3_canonical_50k"]
     + [f"feedback_fractal2_mess3_canonical_s{s}_50k" for s in (1001, 1002, 1003, 1004)]),
    ("loop_mess3_fractal2", ["feedback_mess3_fractal2_canonical_50k"]
     + [f"feedback_mess3_fractal2_canonical_s{s}_50k" for s in (1301, 1302, 1303, 1304)]),
    ("loop_sns_mess3", iv.SETUPS["B (T=sns, U=mess3)"]),
    ("loop_mess3_sns", iv.SETUPS["A (T=mess3, U=sns)"]),
    ("telephone_mess3", it.FAMILIES["telephone"][:1]),
    ("telephone_fractal2", it.FAMILIES["fractal2_telephone"][:1]),
]


def proj_out(D, basis_cols):
    """Remove from each row of D its orthogonal projection onto the column span
    of `basis_cols` ([d_model, k])."""
    if basis_cols.shape[1] == 0:
        return D
    return D - (D @ basis_cols) @ np.linalg.pinv(basis_cols)


def overlap(A, B):
    """Mean squared principal cosine between two column spans (0 = orthogonal,
    1 = one contains the other) -- the project's usual subspace overlap."""
    def basis(M):
        u, s, _ = np.linalg.svd(M, full_matrices=False)
        return u[:, s > s[0] * max(M.shape) * np.finfo(M.dtype).eps]
    Qa, Qb = basis(A), basis(B)
    sv = np.clip(np.linalg.svd(Qa.T @ Qb, compute_uv=False), 0, 1)
    return float((sv ** 2).sum() / max(min(Qa.shape[1], Qb.shape[1]), 1))


def analyze(run_name: str) -> dict:
    P = skc.prepare(run_name)
    E, W = P["E"], P["W"]
    editable, bt, bd = P["editable"], P["bt"], P["bd"]
    process, model, inputs = P["process"], P["model"], P["inputs"]
    base, tstar, hook = P["base"], P["tstar"], P["hook"]

    p_clean = iv.model_dist(model, inputs, tstar, base)
    H_orig = iv.bayes_dist(process, tuple(bt))

    out = {"run": run_name, "editable": editable, "factors": []}
    for f in editable:
        db = bd[f] - bt[f]
        raw = db @ E[f]                                  # unorthogonalized write
        others = [g for g in editable if g != f]
        Wo = np.concatenate([W[g] for g in others], axis=1) if others \
            else np.zeros((raw.shape[1], 0))
        Eo = np.concatenate([E[g].T for g in others], axis=1) if others \
            else np.zeros((raw.shape[1], 0))

        state = list(bt); state[f] = bd[f]
        H_mix = iv.bayes_dist(process, tuple(state))
        m = 0.5 * np.abs(H_orig - H_mix).sum(-1) > TV_FLOOR
        nrm = np.sqrt((db ** 2).sum(-1))
        keep = m & (nrm > 1e-12)
        null = float(np.mean(iv.kl(p_clean, H_mix)[keep]))

        rec = {"factor": f, "n_eval": int(keep.sum()),
               "probe_encoder_overlap": overlap(W[f], E[f].T),
               # between-factor decoder overlap: how much of factor f's encoder
               # span lies in the partners' -- the geometric quantity that
               # decides how much the encoder projection removes
               "decoder_overlap": overlap(E[f].T, Eo) if others else 0.0,
               "null": null,
               "baseline": float(np.mean(iv.kl(p_clean, H_orig)[keep])),
               "schemes": {}}
        for scheme in SCHEMES:
            D = raw if scheme == "none" else \
                proj_out(raw, Wo) if scheme == "probe" else proj_out(raw, Eo)
            rn = np.linalg.norm(raw[keep], axis=-1)
            removed = float(np.mean(
                np.linalg.norm((raw - D)[keep], axis=-1) / np.maximum(rn, 1e-30)))
            lam = float(np.mean(((D @ W[f])[keep] * db[keep]).sum(-1)
                                / (nrm[keep] ** 2)))
            leak_probe = float(np.mean(np.linalg.norm((D @ Wo)[keep], axis=-1)
                                       / nrm[keep])) if others else 0.0
            enc_hit = D - proj_out(D, Eo) if others else np.zeros_like(D)
            leak_enc = float(np.mean(np.linalg.norm(enc_hit[keep], axis=-1)
                                     / np.maximum(np.linalg.norm(D[keep], axis=-1),
                                                  1e-30))) if others else 0.0
            p_patched = iv.model_dist(model, inputs, tstar, base, edit=(hook, D))
            measured = float(np.mean(iv.kl(p_patched, H_mix)[keep]))
            rec["schemes"][scheme] = {
                "removed": removed, "lambda": lam, "leak_probe": leak_probe,
                "leak_enc": leak_enc, "measured": measured,
                "gap_closed": 1.0 - measured / max(null, 1e-12)}
        out["factors"].append(rec)
    return out


def main() -> None:
    res = []
    for group, runs in GROUPS:
        print(f"== {group}", flush=True)
        for run in runs:
            r = analyze(run)
            r["group"] = group
            res.append(r)
            for fac in r["factors"]:
                s = fac["schemes"]
                print(f"   {run[-26:]:26s} f{fac['factor']} "
                      f"W/E-overlap {fac['probe_encoder_overlap']:.3f} | " +
                      " | ".join(
                          f"{k}: rm {s[k]['removed']:.4f} gap {s[k]['gap_closed']:+.3f}"
                          for k in SCHEMES), flush=True)
    (HERE / "steering_orth_variants.json").write_text(json.dumps(res, indent=2))

    print("\n=== per group: gap closed by orthogonalization scheme ===")
    print(f"{'group':22s} {'W/E ovl':>8s} " +
          " ".join(f"{k:>18s}" for k in SCHEMES))
    for group, _ in GROUPS:
        facs = [f for r in res if r["group"] == group for f in r["factors"]]
        if not facs:
            continue
        ov = np.mean([f["probe_encoder_overlap"] for f in facs])
        cells = []
        for k in SCHEMES:
            g = np.mean([f["schemes"][k]["gap_closed"] for f in facs])
            rm = np.mean([f["schemes"][k]["removed"] for f in facs])
            cells.append(f"{g:+.3f} (rm {rm:.3f})")
        print(f"{group:22s} {ov:8.3f} " + " ".join(f"{c:>18s}" for c in cells))


if __name__ == "__main__":
    main()
