"""Shared primitives of the belief-patching pipelines.

Bayes references from the generative process (`bayes_dist`), the model's
next-token distribution with an optional residual-stream edit at the last
trained position (`model_dist`), rowwise KL / total variation, and the two
sns feedback families (`SETUPS`). Used by steering_kl_controls_paper.py,
steering_orth_variants.py and the dose-response pipeline.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import jax

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

from plotting.lib.runs import run_path  # noqa: E402

__all__ = ["run_path", "SETUPS", "TV_FLOOR", "EPS", "kl", "tv", "model_dist", "bayes_dist"]
SETUPS = {
    # run names are feedback_<T>_<U>; setup A has mess3 as T (as in the paper)
    "A (T=mess3, U=sns)": ["feedback_mess3_sns_canonical_50k"]
    + [f"feedback_mess3_sns_canonical_s{s}_50k" for s in (201, 202, 203, 204)],
    "B (T=sns, U=mess3)": ["feedback_sns_mess3_canonical_50k"]
    + [f"feedback_sns_mess3_canonical_s{s}_50k" for s in (101, 102, 103, 104)],
}
TV_FLOOR = 0.05          # identifiability: references must differ by this much
EPS = 1e-12


def kl(p, q):  # rowwise KL(p||q), [n, V]
    p = np.clip(p, EPS, None); q = np.clip(q, EPS, None)
    return (p * (np.log(p) - np.log(q))).sum(-1)


def tv(p, q):
    return 0.5 * np.abs(p - q).sum(-1)


def model_dist(model, inputs, tstar, base_vocab, edit=None):
    """Next-token distribution at position tstar (renormalized over base vocab).
    edit: (layer_hookname, delta[B, d_model]) added at tstar."""
    if edit is None:
        with torch.no_grad():
            logits = model(inputs)
    else:
        name, delta = edit
        d = torch.as_tensor(delta, dtype=torch.float32)

        def hook(act, hook):
            act[:, tstar, :] = act[:, tstar, :] + d
            return act
        with torch.no_grad():
            logits = model.run_with_hooks(inputs, fwd_hooks=[(name, hook)])
    lg = logits[:, tstar, :base_vocab].double()
    return torch.softmax(lg, dim=-1).cpu().numpy()


def bayes_dist(process, state_tuple):
    fn = process.observation_probability_distribution
    return np.asarray(jax.vmap(fn)(tuple(jax.numpy.asarray(s) for s in state_tuple)))
