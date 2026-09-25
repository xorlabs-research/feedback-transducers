"""Shared loaders + probe/geometry helpers for the feedback-transducer figures.

The notebooks and the analysis scripts share this one implementation; the
paper figures were generated with these exact code paths.

Run dirs are resolved through :mod:`plotting.lib.runs` (honors the
``FACTORED_REPS_DATA`` env var), and contain ``hydra/config.yaml`` +
``checkpoints/<step>/model.pt``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import jax
import hydra
from omegaconf import OmegaConf


# Probe-batch conventions shared by every figure: one fixed batch of BATCH
# sequences sampled with PRNGKey(SEED), so theory references are identical
# across runs/seeds.
CKPT_FINAL = 50000
BATCH = 1024
SEED = 12345

# Agent-U emission alphabet V by process name (the latched-action alphabet X of
# the action-delayed factor is V of its first stacked HMM).
V_TABLE = {"sns": 2, "mess3": 3, "coin": 2, "fractal2": 2, "even_ones": 2,
           "no_consecutive_ones": 2, "zero_one_random": 2, "rrxor": 2,
           "tom_quantum": 4}


def load_cfg_proc(run_dir: Path):
    if not OmegaConf.has_resolver("hydra"):
        OmegaConf.register_new_resolver("hydra", lambda key: f"<hydra:{key}>")
    cfg = OmegaConf.load(Path(run_dir) / "hydra" / "config.yaml")
    cfg.device = "cpu"
    process = hydra.utils.instantiate(cfg.generative_process.instance)
    base_vocab = int(process.vocab_size)
    vocab_size = base_vocab + 1
    cfg.generative_process.vocab_size = vocab_size
    cfg.generative_process.base_vocab_size = base_vocab
    cfg.generative_process.bos_token = base_vocab
    cfg.predictive_model.instance.cfg.d_vocab = vocab_size
    cfg.predictive_model.instance.cfg.device = "cpu"
    return cfg, process


def load_model(cfg, ckpt_path: Path):
    model = hydra.utils.instantiate(cfg.predictive_model.instance)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    elif isinstance(state, dict):
        model.load_state_dict(state)
    model.eval()
    return model


def final_ckpt(run_dir: Path) -> Path:
    """Highest-numbered checkpoint (runs differ in length; never hardcode)."""
    run_dir = Path(run_dir)
    steps = sorted(int(p.name) for p in (run_dir / "checkpoints").iterdir()
                   if p.name.isdigit() and (p / "model.pt").exists())
    if not steps:
        raise FileNotFoundError(f"no checkpoints under {run_dir}")
    return run_dir / f"checkpoints/{steps[-1]}/model.pt"


def sample_batch(cfg, process, batch: int = BATCH, seed: int = SEED):
    from fwh_core.generative_processes.torch_generator import (
        generate_data_batch_with_full_history,
    )
    from experiments.training.utils import expand_state_by_batch_size

    bos = cfg.generative_process.bos_token
    eos = cfg.generative_process.eos_token
    n_ctx = cfg.predictive_model.instance.cfg.n_ctx
    seq_len = n_ctx - int(bos is not None) - int(eos is not None)
    init = expand_state_by_batch_size(process.initial_state, batch)
    outs = generate_data_batch_with_full_history(
        init, process, batch, seq_len,
        jax.random.PRNGKey(seed),
        bos_token=bos, eos_token=eos,
    )
    return outs["inputs"], outs["belief_states"]


def acts_at_last_resid_post(model, inputs):
    with torch.no_grad():
        _, cache = model.run_with_cache(inputs, return_type="logits")
    keys = sorted(k for k in cache.keys() if "resid_post" in k)
    return cache[keys[-1]].cpu().numpy()


def cev_of_rows(X):
    Xc = X - X.mean(axis=0, keepdims=True)
    s = np.linalg.svd(Xc, compute_uv=False)
    var = s ** 2
    return np.cumsum(var) / max(var.sum(), 1e-30)


def stack_beliefs(beliefs):
    if not isinstance(beliefs, tuple):
        beliefs = (beliefs,)
    return [np.asarray(b) for b in beliefs]


def kron_per_token(b_list):
    out = b_list[0]
    for b in b_list[1:]:
        out = np.einsum("bti,btj->btij", out, b).reshape(*out.shape[:-1], -1)
    return out


def fit_predict_linear(X_tr, Y_tr, X_full):
    X1 = np.hstack([X_tr, np.ones((X_tr.shape[0], 1))])
    sol, *_ = np.linalg.lstsq(X1, Y_tr, rcond=None)
    W, b = sol[:-1], sol[-1]
    return X_full @ W + b


def emission_marginal(b_aug, action_size):
    last = b_aug.shape[-1]
    S = last // action_size
    bel = b_aug.reshape(*b_aug.shape[:-1], S, action_size)
    return bel.sum(axis=-2)  # sum over s


def color_from_marginal(p):
    V = p.shape[-1]
    if V == 2:
        x = np.clip(p[..., 0], 0, 1)
        return np.stack([x, np.zeros_like(x), 1.0 - x], axis=-1)
    return np.clip(p, 0, 1)


def pca_basis(rows, k=2):
    Xc = rows - rows.mean(axis=0, keepdims=True)
    _u, _s, Vt = np.linalg.svd(Xc, full_matrices=False)
    return rows.mean(axis=0), Vt[:k].T


def project(rows, mu, basis):
    return (rows - mu) @ basis


def u_action_size(cfg):
    """For feedback configs agent U is the 2nd factor; its emission alphabet
    (= latched-action alphabet X) is V of its first stacked HMM."""
    factors = cfg.generative_process.instance.get("factors")
    u = factors[1]
    pn = u["transducer"]["from_hmms"][0]["process_name"]
    return V_TABLE[pn]
