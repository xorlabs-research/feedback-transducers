"""Dose-response curve for belief steering in the mess3<->fractal2 feedback loops.

The paper's belief-patching result (fig:feedback_msp panels i,j) is an ENDPOINT
measurement: splice a complete counterfactual belief b'_f into one factor and
check that the output moves to the Bayes prediction at (b'_f, b_{-f}). This
script interpolates the edit,

    b_f(lambda) = (1 - lambda) b_f + lambda b_f' ,   lambda in {0, 1/8, ..., 1},

(a convex combination of two points of the belief simplex, hence itself a valid
belief) and asks whether the model's output tracks the Bayes prediction at the
INTERMEDIATE tuple (b_f(lambda), b_{-f}) continuously along the path. Those
tuples are doubly off-distribution: not only is the spliced pair not jointly
realizable, the interpolated belief is in general not a reachable mixed state of
the factor at all. If the output tracks Bayes at every lambda, the residual
stream's belief code is causally load-bearing as a CONTINUUM, not just at the
two endpoints.

Everything about the write, the references and the gating is reused from the
paper's own machinery -- `steering_kl_controls_paper.prepare` (model, batch,
jointly-fitted decoders E_f / read probes W_f, per-run donor permutation, and
the main-text UNPROJECTED write Delta = E_f (b_f(lambda) - b_f)),
`intervention_factorization.{model_dist,bayes_dist,kl,tv}` for the forward pass
and the exact Bayes references, and `intervention_marginal_readout` for the
per-factor components. Nothing about the intervention is re-derived here.

Per lambda, per patched factor, on the JOINT next-token distribution (nats):

  on_path    KL(p_edited(lambda) || Bayes(b_f(lambda), b_{-f}))   the headline
  to_orig    KL(p_edited(lambda) || Bayes(b_f        , b_{-f}))   lambda=0 target
  to_full    KL(p_edited(lambda) || Bayes(b_f'       , b_{-f}))   lambda=1 target
  floor      KL(p_clean          || Bayes(b_f        , b_{-f}))   model's own
                                                                  fidelity

plus two per-factor component reads (marginal read-out of the same output):

  leak       KL of the UNTOUCHED factor's component against its ORIGINAL
             reference -- the shielding / leakage check across lambda
  own        KL of the PATCHED factor's component against its lambda target

Identifiability gating is the paper's: an intervention counts only if the two
ENDPOINT references are separated by more than TV_FLOOR = 0.05, so the evaluated
subset is exactly the one the endpoint result is reported on.

Writes dose_response.json (summaries) and dose_response.npz (every
per-intervention value).

Run from repo root:
    python experiments/fig17_dose_response/dose_response.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "analysis"))

import intervention_factorization as iv          # noqa: E402  (analysis/)
import intervention_marginal_readout as mr       # noqa: E402
import steering_kl_controls_paper as kc          # noqa: E402

TV_FLOOR = kc.TV_FLOOR                            # 0.05, the paper's gate
LAMBDAS = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]

# (setup key, label, V_T (agent T's emission alphabet), t_cond, runs)
# V_T / t_cond follow the paper's rule (conditional T read-out iff
# V_T > V_U). Setup letters: A = the mess3-as-T loop (T = mess3), B = the
# role-swapped loop.
SETUPS = [
    ("loop_fractal2_mess3", "B:  $T$ = fractal2,  $U$ = mess3", 2, False,
     ["feedback_fractal2_mess3_canonical_50k"]
     + [f"feedback_fractal2_mess3_canonical_s{s}_50k"
        for s in (1001, 1002, 1003, 1004)]),
    ("loop_mess3_fractal2", "A:  $T$ = mess3,  $U$ = fractal2", 3, True,
     ["feedback_mess3_fractal2_canonical_50k"]
     + [f"feedback_mess3_fractal2_canonical_s{s}_50k"
        for s in (1301, 1302, 1303, 1304)]),
]
FACTOR_NAMES = {0: "agent T", 1: "agent U"}


def analyze(run_name: str, v_t: int, t_cond: bool,
            orth: str = "none") -> dict:
    """Full lambda sweep for one run, both patched factors.

    orth: "none" -> the main text's unprojected write Delta (kc.prepare's own);
          "encoder" -> Delta_perp of eq:steering_write, orthogonalized against
          the partner's ENCODER span, as the appendix's variant. Robustness
          check only; the headline uses "none".
    """
    P = kc.prepare(run_name)
    process, model, inputs = P["process"], P["model"], P["inputs"]
    base, tstar, hook = P["base"], P["tstar"], P["hook"]
    bt, bd = P["bt"], P["bd"]
    if orth == "none":
        write_delta = P["write_delta"]
    else:
        E = P["E"]

        def write_delta(i, db, _E=E):
            delta = db @ _E[i]
            basis = np.concatenate([_E[j].T for j in _E if j != i], axis=1)
            return delta - (delta @ basis) @ np.linalg.pinv(basis)
    v_u = base // v_t
    assert v_t * v_u == base, f"{run_name}: V_T={v_t} does not divide {base}"
    assert len(bt) == 2, f"{run_name}: expected 2 factors"

    p_clean = iv.model_dist(model, inputs, tstar, base)
    H_orig = iv.bayes_dist(process, tuple(bt))
    floor_all = iv.kl(p_clean, H_orig)

    def comps(b0, b1):
        return mr.comps_wrap(process, b0, b1, v_t, v_u, t_cond)

    t_ref, w_ref, u_ref = comps(bt[0], bt[1])

    out = {"run": run_name, "vocab": base, "V": [v_t, v_u], "orth": orth,
           "t_cond": t_cond, "dims": [b.shape[-1] for b in bt],
           "editable": P["editable"], "lambdas": LAMBDAS, "factors": {}}
    raw = {}

    for f in P["editable"]:
        state = list(bt)
        state[f] = bd[f]
        H_full = iv.bayes_dist(process, tuple(state))
        # the paper's identifiability gate, on the ENDPOINT references
        m = iv.tv(H_orig, H_full) > TV_FLOOR
        n_eval = int(m.sum())

        rows = {k: [] for k in ("on_path", "to_orig", "to_full", "null",
                                "leak", "own")}
        for lam in LAMBDAS:
            b_lam = (1.0 - lam) * bt[f] + lam * bd[f]
            st = list(bt)
            st[f] = b_lam
            H_lam = iv.bayes_dist(process, tuple(st))
            delta = write_delta(f, b_lam - bt[f])
            p_lam = iv.model_dist(model, inputs, tstar, base,
                                  edit=(hook, delta))

            rows["on_path"].append(iv.kl(p_lam, H_lam)[m])
            rows["to_orig"].append(iv.kl(p_lam, H_orig)[m])
            rows["to_full"].append(iv.kl(p_lam, H_full)[m])
            # the paper's null for THIS lambda: what an inert write would give
            # against the same target. Makes the paper's headline statistic,
            # closed gap = 1 - measured/null, available at every dose.
            rows["null"].append(iv.kl(p_clean, H_lam)[m])

            # per-factor component read-out of the SAME edited output
            m_t, w_m = mr.t_component(p_lam, v_t, v_u, t_cond)
            m_u = p_lam.reshape(-1, v_t, v_u).sum(1)
            t_lam, w_lam, u_lam = comps(*st)          # lambda-target comps
            if f == 0:      # patched T, untouched U
                rows["own"].append(mr.comp_kl(m_t, t_lam, w_m)[m])
                rows["leak"].append(mr.comp_kl(m_u, u_ref, None)[m])
            else:           # patched U, untouched T
                rows["own"].append(mr.comp_kl(m_u, u_lam, None)[m])
                rows["leak"].append(mr.comp_kl(m_t, t_ref, w_m)[m])

        arrs = {k: np.stack(v) for k, v in rows.items()}   # [n_lambda, n_eval]
        arrs["floor"] = floor_all[m][None, :]
        for k, a in arrs.items():
            raw[f"{run_name}|{f}|{k}"] = a.astype(np.float32)

        def summ(a):
            return {"median": [float(x) for x in np.median(a, 1)],
                    "q25": [float(x) for x in np.quantile(a, 0.25, 1)],
                    "q75": [float(x) for x in np.quantile(a, 0.75, 1)],
                    "mean": [float(x) for x in np.mean(a, 1)]}

        rec = {"n_eval": n_eval, "n_total": int(m.size)}
        rec.update({k: summ(a) for k, a in arrs.items()})
        out["factors"][str(f)] = rec

        print(f"     factor {f} ({FACTOR_NAMES[f]}, n={n_eval}): "
              f"floor {rec['floor']['median'][0]:.2e} | on-path med "
              + " ".join(f"{x:.2e}" for x in rec["on_path"]["median"]),
              flush=True)
    return out, raw


def main() -> None:
    t0 = time.time()
    orth = "encoder" if "--orth" in sys.argv else "none"
    stem = "dose_response_orth" if orth != "none" else "dose_response"
    res, raw_all = [], {}
    for key, label, v_t, t_cond, runs in SETUPS:
        print(f"== {key}: {len(runs)} run(s)", flush=True)
        for run in runs:
            print(f"   {run}", flush=True)
            r, raw = analyze(run, v_t, t_cond, orth)
            r["setup"], r["label"] = key, label
            res.append(r)
            raw_all.update(raw)
    (HERE / f"{stem}.json").write_text(json.dumps(
        {"lambdas": LAMBDAS, "tv_floor": TV_FLOOR, "orth": orth,
         "write": ("unprojected (main text, eq:steering_write Delta)"
                   if orth == "none"
                   else "encoder-orthogonalized (eq:steering_write Delta_perp)"),
         "setups": [{"key": k, "label": l, "V_T": v, "t_cond": c,
                     "runs": rr} for k, l, v, c, rr in SETUPS],
         "runs": res}, indent=2))
    np.savez_compressed(HERE / f"{stem}.npz", **raw_all)
    print(f"\nwrote {stem}.json + {stem}.npz "
          f"({len(res)} runs, {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
