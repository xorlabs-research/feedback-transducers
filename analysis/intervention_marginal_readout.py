"""Per-factor read-out helpers for the belief-patching pipelines.

Decomposes the model's joint next-token prediction into per-factor
components (`t_component`, `comps_wrap`), compares them to Bayes references
in TV or KL (`comp_dist`, `comp_kl`, `classify_component`), and provides the
order-independent per-run donor permutation (`donor_rng`) that every
patching analysis draws its counterfactual beliefs from.
"""
from __future__ import annotations

import sys
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from plotting.lib import feedback_runs as ft      # noqa: E402
import intervention_factorization as iv          # noqa: E402

TV_FLOOR = iv.TV_FLOOR
DONOR_SEED = 7
# The donor permutation is derived per run name, so results do not depend on
# call order.


def donor_rng(run_name: str) -> np.random.Generator:
    """Deterministic per-run donor generator (order-independent)."""
    return np.random.default_rng(DONOR_SEED + zlib.crc32(run_name.encode()))


def t_component(P, V_t, V_u, cond):
    """P: [n, base] joint. Returns (component, weights) — marginal P(x) or the
    stack of conditionals {P(x|y)}_y with weights P(y)."""
    M = P.reshape(-1, V_t, V_u)
    if not cond:
        return M.sum(2), None                     # [n, V_t]
    Py = M.sum(1)                                 # [n, V_u]
    Pxy = M / np.clip(M.sum(1, keepdims=True), iv.EPS, None)   # P(x|y)
    return Pxy, Py                                # [n, V_t, V_u], [n, V_u]


def comp_dist(A, B, w=None):
    """TV between components; for conditionals, expectation over y under w."""
    if w is None:
        return 0.5 * np.abs(A - B).sum(-1)
    return (w * 0.5 * np.abs(A - B).sum(1)).sum(-1)


def comp_kl(A, B, w=None):
    """KL(A||B) between components; for conditionals, expectation over y under
    w. Same weighting convention as comp_dist, so the two metrics are
    interchangeable in classify_component."""
    a = np.clip(A, iv.EPS, None)
    b = np.clip(B, iv.EPS, None)
    d = (a * (np.log(a) - np.log(b)))
    if w is None:
        return d.sum(-1)
    return (w * d.sum(1)).sum(-1)


def comps_wrap(process, b_t, b_u, V_t, V_u, cond):
    """Exact per-factor components of the Bayes prediction at (b_t, b_u)."""
    P = iv.bayes_dist(process, (b_t, b_u))
    t_c, w = t_component(P, V_t, V_u, cond)
    u_c = P.reshape(-1, V_t, V_u).sum(1)
    return t_c, w, u_c


BROKEN_TV = 0.25         # fixed TV threshold (the paper figures)


def classify_component(comp, w, ref_o, ref_c, metric="tv", eps=None):
    """Label each component 'orig'/'cf' by nearest reference, 'broken' if far
    from both; `ident` flags rows whose references are separated enough to be
    distinguishable at all.

    metric: 'tv' -> total variation, threshold BROKEN_TV;
            'kl' -> KL(component||reference) in nats, threshold `eps`
                    (calibrated per run on the UNEDITED output).
    The identifiability gate always uses TV, so both metrics are evaluated on
    exactly the same subset of interventions."""
    dist = comp_dist if metric == "tv" else comp_kl
    d_o = dist(comp, ref_o, w)
    d_c = dist(comp, ref_c, w)
    ident = comp_dist(ref_o, ref_c, w) > TV_FLOOR
    lab = np.where(d_c < d_o, "cf", "orig").astype(object)
    thr = BROKEN_TV if eps is None else eps
    lab[np.minimum(d_o, d_c) > thr] = "broken"
    return lab, ident, d_o, d_c


# single definition of the final-checkpoint lookup lives in the plotting library
final_ckpt = ft.final_ckpt
