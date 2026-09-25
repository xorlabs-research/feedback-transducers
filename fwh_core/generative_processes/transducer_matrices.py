"""Transducer transition matrices for feedback-transducer feedback processes.

Each function returns a tensor of shape `[K, V, S, S]` where K is the input
alphabet size and V the output alphabet size. These are the per-factor
matrices consumed by `feedback_transducer.build_feedback_transducer_process_from_spec`.

Convention: T[k, v, s, s'] = Pr(emit v, new state s' | input k, current state s).
Sum over (v, s') for fixed (k, s) is 1.

The transducers in this module are role-agnostic: any of them may serve as
the driven (agent T) factor or as a delayed (agent U) factor in a coupled loop,
depending on how the spec wires inputs and outputs.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp

from fwh_core.generative_processes.transition_matrices import mess3


def binary_flip(p: float = 0.8) -> jax.Array:
    """Two-state binary controlled-flip transducer.

    State and emissions are 2-valued. Input 0 stays in state with prob p
    (and emits the resulting state); input 1 flips with prob p. Emission
    always equals the new (post-transition) state.

    Shape: [K=2, V=2, S=2, S=2].
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1], got {p}")
    T = jnp.zeros((2, 2, 2, 2))
    # input 0: stay with prob p
    T = T.at[0, 0, 0, 0].set(p)
    T = T.at[0, 1, 0, 1].set(1 - p)
    T = T.at[0, 0, 1, 0].set(1 - p)
    T = T.at[0, 1, 1, 1].set(p)
    # input 1: flip with prob p
    T = T.at[1, 0, 0, 0].set(1 - p)
    T = T.at[1, 1, 0, 1].set(p)
    T = T.at[1, 0, 1, 0].set(p)
    T = T.at[1, 1, 1, 1].set(1 - p)
    return T


def copy(alphabet_size: int = 2) -> jax.Array:
    """Memoryless deterministic copy: output equals input.

    Shape: [K=alphabet_size, V=alphabet_size, S=1, S=1].
    """
    if alphabet_size < 1:
        raise ValueError(f"alphabet_size must be >= 1, got {alphabet_size}")
    U = jnp.zeros((alphabet_size, alphabet_size, 1, 1))
    for k in range(alphabet_size):
        U = U.at[k, k, 0, 0].set(1.0)
    return U


def lookup(table: Sequence[int], output_size: int) -> jax.Array:
    """Memoryless deterministic lookup: output `v = table[k]`.

    Args:
        table: list of length `input_size`, each entry an int in
            `[0, output_size)`.
        output_size: emission alphabet size.

    Shape: [K=len(table), V=output_size, S=1, S=1].
    """
    input_size = len(table)
    if any(int(t) < 0 or int(t) >= output_size for t in table):
        raise ValueError(f"table entries must be in [0, {output_size})")
    U = jnp.zeros((input_size, output_size, 1, 1))
    for k, v in enumerate(table):
        U = U.at[k, int(v), 0, 0].set(1.0)
    return U


def epsilon_greedy_lookup(
    table: Sequence[int],
    output_size: int,
    epsilon: float,
) -> jax.Array:
    """Epsilon-greedy lookup transducer.

    With probability `1 - epsilon`, emit `table[input]` (deterministic
    lookup). With probability `epsilon`, emit a uniform-random output.
    Memoryless. Special cases: `epsilon=0` reduces to `lookup(table)`;
    `epsilon=1` is uniform-random emission, independent of input.

    Args:
        table: list of length `input_size`, each entry in `[0, output_size)`.
        output_size: emission alphabet size.
        epsilon: exploration probability in `[0, 1]`.

    Shape: [K=len(table), V=output_size, S=1, S=1].
    """
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError(f"epsilon must be in [0, 1], got {epsilon}")
    if any(int(t) < 0 or int(t) >= output_size for t in table):
        raise ValueError(f"table entries must be in [0, {output_size})")
    input_size = len(table)
    table_arr = jnp.asarray([int(t) for t in table], dtype=jnp.int32)
    onehot = jax.nn.one_hot(table_arr, output_size)               # [K, V]
    uniform = jnp.full((input_size, output_size), 1.0 / output_size)
    probs = (1.0 - epsilon) * onehot + epsilon * uniform          # [K, V]
    return probs[:, :, None, None]


def input_driven_mess3(x: float, a: float, beta: float) -> jax.Array:
    """Input-driven mess3 transducer.

    Wraps the 3-state, 3-emission mess3 process with an input-conditioned
    state rotation. Input alphabet K = {0, 1, 2}; for each input k:

        T^(y|k) = R_k . T^(y),  R_k = (1 - beta) I + beta P^k

    where P is the cyclic permutation A -> B -> C -> A on the 3 hidden
    states. Input 0 (P^0 = I) leaves the state alone; input 1 partially
    rotates forward; input 2 partially rotates backward (P^2 = P^{-1}
    on a 3-cycle). Emission probabilities given the post-rotation state
    follow base mess3 unchanged.

    Args:
        x: mess3 emission-bias parameter.
        a: mess3 stay-probability parameter.
        beta: rotation strength in [0, 1]. beta=0 collapses all three
            inputs to identical base mess3; beta=1 fully rotates per P^k.

    Shape: [K=3, V=3, S=3, S=3].
    """
    if not 0.0 <= beta <= 1.0:
        raise ValueError(f"beta must be in [0, 1], got {beta}")
    base = mess3(x=x, a=a)                                 # [V=3, S=3, S=3]
    P = jnp.array(
        [[0, 1, 0],
         [0, 0, 1],
         [1, 0, 0]],
        dtype=base.dtype,
    )
    I = jnp.eye(3, dtype=base.dtype)
    P2 = P @ P
    R = jnp.stack(
        [I, (1 - beta) * I + beta * P, (1 - beta) * I + beta * P2],
        axis=0,
    )                                                      # [3, S, S]
    # T_input[k, v, s, s'] = sum_{s''} R[k, s, s''] . base[v, s'', s']
    return jnp.einsum("akl,vlm->avkm", R, base)


TRANSDUCER_MATRIX_FUNCTIONS = {
    "binary_flip": binary_flip,
    "copy": copy,
    "lookup": lookup,
    "epsilon_greedy_lookup": epsilon_greedy_lookup,
    "input_driven_mess3": input_driven_mess3,
}
