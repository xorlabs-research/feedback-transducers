"""Compute position-specific Bayes-optimal loss for generative processes.

This module provides the core function for computing E[-log P(X_t | X_{1:t-1})]
for each position t, which represents the theoretical minimum achievable loss.
"""

from typing import Any

import jax
import jax.numpy as jnp

from fwh_core.generative_processes.generative_process import GenerativeProcess

from utils import expand_state_by_batch_size


def _token_probs_chunked(generative_process, state_t, tokens_t, num_samples, chunk):
    """P(actual token | state) per sample, chunked over the sample axis.

    Numerically identical to `vmap(observation_probability_distribution)` over
    all `num_samples` followed by a gather, but never materializes the full
    [num_samples, vocab] observation-probability array. That matters twice:
    (a) the full-batch vmap fuses the per-agent joint contraction into one giant
    [K_t, num_samples * partial, NAGS] buffer, which OOMs for high-factor
    processes (e.g. the 5-agent fractal2 telephone tuple); (b) for large joint
    vocabularies even the *output* array explodes — 32768 samples x 59049 vocab
    (5-agent mess3 telephone) = 7.2 GiB. Gathering the
    actual token's probability inside each chunk keeps the peak at
    [chunk, vocab].
    """
    fn = generative_process.observation_probability_distribution

    def _slice(lo, hi):
        if isinstance(state_t, tuple):
            return tuple(b[lo:hi] for b in state_t)
        return state_t[lo:hi]

    parts = []
    for lo in range(0, num_samples, chunk):
        hi = min(lo + chunk, num_samples)
        probs = jax.vmap(fn)(_slice(lo, hi))          # [c, vocab]
        parts.append(probs[jnp.arange(hi - lo), tokens_t[lo:hi]])
    return jnp.concatenate(parts, axis=0)             # [num_samples]


def compute_position_optimal_loss(
    generative_process: GenerativeProcess,
    num_samples: int,
    sequence_len: int,
    seed: int,
    *,
    compute_stderr: bool = False,
    obs_chunk_size: int = 2048,
) -> dict[str, Any]:
    """Compute position-specific Bayes-optimal loss.

    For each position t in [0, sequence_len-1], computes:
        E[-log P(X_{t+1} | X_{0:t})]

    where the expectation is over sequences X sampled from the generative process.

    Args:
        generative_process: The generative process to analyze
        num_samples: Number of sequences to sample for Monte Carlo estimation
        sequence_len: Length of sequences to generate
        seed: Random seed
        compute_stderr: Whether to compute standard errors (slightly slower)

    Returns:
        Dictionary with:
            - per_position_loss: list of E[-log P(X_{t+1}|X_{0:t})] for t=0,...,seq_len-1
            - per_position_stderr: list of standard errors (only if compute_stderr=True)
            - average_loss: scalar average over all positions
            - asymptotic_estimate: loss at final position (closest to entropy rate)
            - position_1_loss: loss at first position
            - num_positions: number of positions analyzed
    """
    key = jax.random.PRNGKey(seed)

    # Get initial states for the batch
    initial_states = expand_state_by_batch_size(
        generative_process.initial_state, num_samples
    )

    # Generate sequences with full belief state tracking
    batch_keys = jax.random.split(key, num_samples)
    belief_states, tokens = generative_process.generate(
        initial_states, batch_keys, sequence_len, True  # return_all_states=True
    )

    # Compute -log P(X_t | belief_state_t) at each position
    per_position_losses = []
    per_position_stderrs = [] if compute_stderr else None

    for t in range(sequence_len):
        # Get belief state at position t (before seeing token t)
        if isinstance(belief_states, tuple):
            state_t = tuple(b[:, t, ...] for b in belief_states)
        else:
            state_t = belief_states[:, t, ...]

        # Probability of the actual token at position t. Chunked over samples
        # to bound both the intermediate einsum and the [samples, vocab]
        # observation-probability array (see _token_probs_chunked); identical
        # result to a single full-batch vmap + gather.
        token_probs = _token_probs_chunked(
            generative_process, state_t, tokens[:, t], num_samples, obs_chunk_size
        )

        # Compute -log P (in nats)
        neg_log_probs = -jnp.log(token_probs + 1e-10)

        # Average over samples
        mean_loss = float(jnp.mean(neg_log_probs))
        per_position_losses.append(mean_loss)

        if compute_stderr and per_position_stderrs is not None:
            stderr = float(jnp.std(neg_log_probs) / jnp.sqrt(num_samples))
            per_position_stderrs.append(stderr)

    average_loss = sum(per_position_losses) / len(per_position_losses)

    result = {
        "per_position_loss": per_position_losses,
        "average_loss": average_loss,
        "asymptotic_estimate": per_position_losses[-1],
        "position_1_loss": per_position_losses[0],
        "num_positions": sequence_len,
    }

    if compute_stderr:
        result["per_position_stderr"] = per_position_stderrs

    return result


def print_optimal_loss_results(results: dict[str, Any], process_name: str) -> None:
    """Print optimal loss results in a formatted way."""
    print(f"\n{'='*60}")
    print(f"Position-Specific Optimal Loss for: {process_name}")
    print(f"{'='*60}")

    print(f"\nNumber of positions: {results['num_positions']}")
    print(f"\nPer-position loss (nats) [E[-log P(X_t | X_{{1:t-1}})]]:")
    print("-" * 50)

    has_stderr = "per_position_stderr" in results
    if has_stderr:
        print(f"{'Position':>10} {'Loss (nats)':>15} {'Stderr':>12}")
    else:
        print(f"{'Position':>10} {'Loss (nats)':>15}")
    print("-" * 50)

    for t, loss in enumerate(results['per_position_loss']):
        if has_stderr:
            stderr = results['per_position_stderr'][t]
            print(f"{t+1:>10} {loss:>15.6f} {stderr:>12.6f}")
        else:
            print(f"{t+1:>10} {loss:>15.6f}")

    print("-" * 50)
    print(f"\n{'Summary Statistics':^50}")
    print("-" * 50)
    print(f"Position 1 loss (first prediction):  {results['position_1_loss']:.6f} nats")
    print(f"Final position loss (asymptotic):    {results['asymptotic_estimate']:.6f} nats")
    print(f"Average loss over all positions:     {results['average_loss']:.6f} nats")
    print(f"Spread (pos 1 - final):              {results['position_1_loss'] - results['asymptotic_estimate']:.6f} nats")
    print("-" * 50)

    # Interpretation
    print(f"\n{'Interpretation':^50}")
    print("-" * 50)
    print("If a model achieves loss close to 'Average loss', it is near-optimal")
    print("for the position-averaged metric used in training.")
    print("")
    print("The 'Final position loss' approximates the asymptotic entropy rate H∞.")
    print("The gap between average and final shows early-position penalty.")
    print("=" * 60)
