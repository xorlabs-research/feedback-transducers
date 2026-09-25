"""Effective rank of the final-layer residual stream vs number of sub-agents N.

The paper's central quantitative claim: the residual stream's belief-geometry
dimension grows ~LINEARLY in the number of agents N (factored bound
$\\sum_i(d_i-1)$) rather than EXPONENTIALLY (joint-Kronecker bound
$\\prod_i d_i - 1$). `effective_rank` measures the empirical effective rank of one trained run,
`aggregate_seeds` pools seeds, `theory_thresholds` gives the same statistic for
the Bayes-optimal beliefs, and `draw_rank_panel` plots the sweep on a log axis.

Effective rank nc_p = smallest #PCs whose cumulative explained variance of the
(BOS-excluded) final-layer resid_post reaches p (default p in {0.80, 0.95}).

The references are held to the SAME thresholds: on the same probe batch, the
n_c@p of the factored concatenation of the Bayes beliefs and of their
Kronecker product (theory_thresholds). The total simplex dimensions
Σ(d_i-1) and Πd_i-1 are computed but not drawn: a belief cloud reaches 80/95%
of its variance well below its simplex dimension, so they are not comparable
to n_c.
Colour encodes the threshold, line style the quantity.

The theory bounds come from each run's own per-factor belief dimensions:
dim_factored = Σ(d_i-1), dim_joint = Πd_i - 1.
"""
from __future__ import annotations

import numpy as np
import torch

from .lib.runs import available_steps, load_run, sample_batch

FINAL_STEP = 50000

# colour = CEV threshold (80% red, 95% blue); line style = quantity
THRESHOLD_COLORS = {0.8: "#c0392b", 0.95: "#1f5fa8"}
_FALLBACK_COLORS = ["#c0392b", "#1f5fa8", "#2e8b57", "#8e44ad"]
MAX_KRON_ROWS = 8192   # Gram-matrix row cap for the Kronecker CEV


def _threshold_color(p, j):
    for k, c in THRESHOLD_COLORS.items():
        if abs(k - p) < 1e-9:
            return c
    return _FALLBACK_COLORS[j % len(_FALLBACK_COLORS)]


def _cev(acts: np.ndarray) -> np.ndarray:
    flat = acts[:, 1:, :].reshape(-1, acts.shape[-1])  # skip BOS
    Xc = flat - flat.mean(0, keepdims=True)
    s = np.linalg.svd(Xc, compute_uv=False)
    var = s ** 2
    return np.cumsum(var / max(var.sum(), 1e-30))


def _nc(cev: np.ndarray, p: float) -> int:
    """Smallest #PCs reaching cumulative variance p (1-indexed)."""
    return int(np.searchsorted(cev, p) + 1)


def _cev_rows(X: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(0, keepdims=True)
    s = np.linalg.svd(Xc, compute_uv=False)
    v = s ** 2
    return np.cumsum(v / max(v.sum(), 1e-30))


def _cev_kron_gram(B_list, idx) -> np.ndarray:
    """CEV of the centred Kronecker-product rows without materialising them:
    the Gram matrix of ⊗_k b_k is the Hadamard product of the per-factor Gram
    matrices, and PCA variances are its centred eigenvalues."""
    G = np.ones((len(idx), len(idx)))
    for b in B_list:
        bb = np.asarray(b[idx], dtype=np.float64)
        G *= bb @ bb.T
    Gc = G - G.mean(0, keepdims=True) - G.mean(1, keepdims=True) + G.mean()
    ev = np.clip(np.linalg.eigvalsh(Gc)[::-1], 0.0, None)
    return np.cumsum(ev / max(ev.sum(), 1e-30))


def theory_thresholds(run: str, batch: int, seed: int, thresholds,
                      max_rows: int = MAX_KRON_ROWS) -> dict:
    """n_c@p of the Bayes-optimal beliefs on the SAME probe batch as
    effective_rank(): factored concatenation and Kronecker product. BOS
    position excluded; the trivial 1-dim router factor is dropped (it carries
    no variance), matching the Σ(d_i-1) convention. The Kronecker CEV uses at
    most max_rows subsampled rows, which caps its resolvable rank at
    max_rows-1 (far above any threshold reached here)."""
    cfg, process, _ = load_run(run)
    outs = sample_batch(cfg, process, batch, seed)
    beliefs = outs["belief_states"]
    if not isinstance(beliefs, tuple):
        beliefs = (beliefs,)
    B = [np.asarray(b)[:, 1:, :] for b in beliefs if int(b.shape[-1]) > 1]
    B = [b.reshape(-1, b.shape[-1]) for b in B]
    n = B[0].shape[0]
    cev_f = _cev_rows(np.concatenate(B, axis=-1))
    idx = (np.arange(n) if n <= max_rows
           else np.random.default_rng(0).choice(n, max_rows, replace=False))
    cev_k = _cev_kron_gram(B, idx)
    return {"nc_factored": {p: _nc(cev_f, p) for p in thresholds},
            "nc_kron": {p: _nc(cev_k, p) for p in thresholds},
            "rows_kron": int(len(idx)), "rows_total": int(n)}


def effective_rank(run: str, batch: int, seed: int, thresholds, step: int | None = None) -> dict:
    step_used = step if step is not None else (
        FINAL_STEP if FINAL_STEP in available_steps(run) else max(available_steps(run)))
    cfg, process, model = load_run(run, step=step_used)
    last = int(cfg.predictive_model.instance.cfg.n_layers) - 1
    hook = f"blocks.{last}.hook_resid_post"
    outs = sample_batch(cfg, process, batch, seed)
    inputs = torch.as_tensor(np.asarray(outs["inputs"]), dtype=torch.long)
    with torch.no_grad():
        _, cache = model.run_with_cache(inputs, return_type="logits")
    cev = _cev(cache[hook].cpu().numpy())
    # Closed-form theory dims from the per-factor belief dims:
    # factored = Σ(d_i-1), joint = Πd_i - 1.
    # Avoids materializing the Kronecker belief (9^N dims) for large N.
    beliefs = outs["belief_states"]
    if not isinstance(beliefs, tuple):
        beliefs = (beliefs,)
    dims = [int(b.shape[-1]) for b in beliefs]
    dim_factored = int(sum(d - 1 for d in dims if d > 1))
    dim_joint = int(np.prod(dims)) - 1
    return {"nc": {p: _nc(cev, p) for p in thresholds},
            "dim_factored": dim_factored, "dim_joint": dim_joint,
            "step": step_used, "preliminary": step_used < FINAL_STEP}


def aggregate_seeds(results: list[dict], thresholds) -> dict:
    """Combine per-seed effective_rank() results for one N: mean ± s.d. of the
    empirical ranks; theory dims / step / preliminary from the first seed
    (identical across seeds by construction)."""
    agg = dict(results[0])
    agg["n_seeds"] = len(results)
    agg["nc"] = {p: float(np.mean([r["nc"][p] for r in results])) for p in thresholds}
    agg["nc_min"] = {p: float(np.min([r["nc"][p] for r in results])) for p in thresholds}
    agg["nc_max"] = {p: float(np.max([r["nc"][p] for r in results])) for p in thresholds}
    return agg


def draw_rank_panel(ax, points, thresholds, title: bool = True, fs: float = 10,
                    compact: bool = False, minor_grid: bool = True):
    """Draw the rank-vs-N panel on `ax`. Shared by the standalone figure and by
    the combined telephone figure, so both stay in step.

    Per threshold p (colour): the residual stream's n_c (solid, filled
    circles, min-max band over seeds) against the n_c of the Bayes beliefs on
    the same batch -- factored concatenation (dashed) and Kronecker product
    (dotted), when the points carry them
    (see theory_thresholds). Two legends: line style says WHAT, colour says
    which threshold.

    compact: smaller legends with short labels for multi-panel figures.
    minor_grid: draw a line at every log minor tick."""
    points = sorted(points, key=lambda x: x[0])
    Ns = np.array([N for N, _ in points])
    res = [r for _, r in points]
    prelim = np.array([r.get("preliminary", False) for r in res])
    have_theory = all("nc_factored" in r and "nc_kron" in r for r in res)
    for j, p in enumerate(thresholds):
        c = _threshold_color(p, j)
        if have_theory:
            yk = np.array([r["nc_kron"][p] for r in res])
            yf = np.array([r["nc_factored"][p] for r in res])
            ax.plot(Ns, yk, ":", color=c, lw=1.7)
            ax.plot(Ns, yf, "--", color=c, lw=1.5)
        y = np.array([r["nc"][p] for r in res])
        ylo = np.array([r.get("nc_min", {}).get(p, v) for r, v in zip(res, y)])
        yhi = np.array([r.get("nc_max", {}).get(p, v) for r, v in zip(res, y)])
        if (yhi > ylo).any():
            ax.fill_between(Ns, ylo, yhi, color=c, alpha=0.18, lw=0)
        ax.plot(Ns, y, "-", color=c, lw=1.8)
        # ms=2.2 matches the CEV panels' step curves (plotting.telephone_cev)
        ax.plot(Ns[~prelim], y[~prelim], "o", color=c, ms=2.2)
        ax.plot(Ns[prelim], y[prelim], "o", mfc="none", mec=c, mew=1.2, ms=3.2)
    for N, r in zip(Ns, res):
        if r.get("preliminary"):
            ymax = max(r["nc"][p] for p in thresholds)
            ax.annotate(f"preliminary:\nstep {r['step']//1000}k (training)",
                        (N, ymax), xytext=(-8, 14), textcoords="offset points",
                        ha="right", fontsize=7.5, color="0.25")
    ax.set_yscale("log")
    ax.set_xlabel("number of sub-agents $N$", fontsize=fs)
    ax.set_ylabel("effective rank $n_c$  (log scale)", fontsize=fs)
    ax.set_xticks(Ns)
    if title:
        ax.set_title("Residual-stream rank tracks the factored beliefs, not the joint\n"
                     "(telephone, mess3 sub-agents)", fontsize=11)
    ax.grid(alpha=0.3, which="both" if minor_grid else "major")
    # legends: style = quantity (upper left), colour = threshold (lower right)
    from matplotlib.lines import Line2D
    fs_leg = fs - (2.4 if compact else 2.0)
    kw = dict(fontsize=fs_leg, framealpha=0.92, handlelength=1.6 if compact else 2.2,
              borderpad=0.25, labelspacing=0.18, handletextpad=0.4,
              borderaxespad=0.25)
    names = (("model", "factored", "Kronecker") if compact else
             ("transformer residual stream", "factored Bayes beliefs",
              "Kronecker Bayes beliefs"))
    style = [Line2D([], [], color="0.3", lw=1.8, marker="o", ms=2.2, label=names[0])]
    if have_theory:
        style += [Line2D([], [], color="0.3", lw=1.5, ls="--", label=names[1]),
                  Line2D([], [], color="0.3", lw=1.7, ls=":", label=names[2])]
    colour = [Line2D([], [], color=_threshold_color(p, j), lw=3.5,
                     label=f"{int(round(100 * p))}%" if compact
                     else f"$n_c$@{int(round(100 * p))}%")
              for j, p in enumerate(thresholds)]
    leg1 = ax.legend(handles=style, loc="upper left", **kw)
    ax.add_artist(leg1)
    ax.legend(handles=colour, loc="lower right", **kw)
    return points

