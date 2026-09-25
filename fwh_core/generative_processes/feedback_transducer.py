"""Helpers for building feedback-transducer feedback processes.

`make_action_delayed` wraps agent U's transducer `U^(x|y)` into the
action-delayed form `U_d^(x|y)` from the tensor-product theorem of "Beliefs in
feedback: Factored representations of interacting agents in
transformers". `build_two_agent_loop` is a convenience builder that produces a
`FactoredGenerativeProcess` for the canonical agent T <-> action-delayed agent
U feedback loop.

State convention for the augmented agent U: `idx = latent_idx * action_size + latched_idx`
(action axis inner). `make_action_delayed_initial_state` builds an initial
belief consistent with that layout from a base latent belief and an initial
action distribution.
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Mapping, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from fwh_core.generative_processes.factored_generative_process import FactoredGenerativeProcess
from fwh_core.generative_processes.structures.feedback_transducer import FeedbackTransducerStructure
from fwh_core.generative_processes.transducer_matrices import (
    TRANSDUCER_MATRIX_FUNCTIONS,
    copy as _copy_transducer,
)
from fwh_core.generative_processes.transition_matrices import HMM_MATRIX_FUNCTIONS
from fwh_core.utils.jnp_utils import resolve_jax_device


def make_action_delayed(U: jax.Array, action_size: int) -> jax.Array:
    """Action-delay wrapper for agent U's transducer.

    Given a base transducer `U` of shape `[K=|Y|, V=|X|, S, S]` representing
    `Pr(emit x, new latent s' | percept y, latent s)`, returns the action-delayed
    tensor `U_d` of shape `[K=|Y|, V=|X|, S*|X|, S*|X|]` representing
    `Pr(emit x, new state (s', z') | percept y, state (s, z))`, where the
    augmented state encodes `(latent, latched_action)` and the emission is
    deterministically the previously-latched action: `U_d[k, v, (s,z), (s',z')]
    = delta_{v,z} U[k, z', s, s']`.

    The augmented-state index uses row-major layout `idx = s * action_size + z`
    (latent outer, latched action inner).
    """
    K, V, S, S2 = U.shape
    if S != S2:
        raise ValueError(f"U must be square in latent dim, got {U.shape}")
    if V != action_size:
        raise ValueError(
            f"action_size ({action_size}) must equal U's emission alphabet ({V})"
        )
    eye = jnp.eye(action_size, dtype=U.dtype)               # eye[v, z] = delta_{v,z}
    # einsum: out[k, v, s, z, sp, zp] = eye[v, z] * U[k, zp, s, sp]
    six = jnp.einsum("vz,kpsq->kvszqp", eye, U)
    return six.reshape(K, V, S * action_size, S * action_size)


def make_action_delayed_initial_state(
    latent_belief: jax.Array,
    action_belief: jax.Array,
) -> jax.Array:
    """Initial state for an action-delayed agent U.

    Returns the outer-product belief `latent_belief x action_belief` flattened
    in the augmented-state layout `idx = latent * action_size + latched`.
    """
    if latent_belief.ndim != 1 or action_belief.ndim != 1:
        raise ValueError("latent_belief and action_belief must be 1-D")
    return jnp.einsum("s,a->sa", latent_belief, action_belief).reshape(-1)


def build_two_agent_loop(
    t_transducer: jax.Array,
    u_transducer: jax.Array,
    t_initial_latent: jax.Array | Sequence[float] | None = None,
    u_initial_latent: jax.Array | Sequence[float] | None = None,
    u_initial_action: jax.Array | Sequence[float] | None = None,
    noise_epsilon: float = 0.0,
    device: str | None = None,
) -> FactoredGenerativeProcess:
    """Build the canonical agent T <-> action-delayed agent U feedback loop.

    Args:
        t_transducer: shape `[K=|X|, V=|Y|, S_t, S_t]`. K is the action
            input alphabet; V is the perception emission alphabet.
        u_transducer: shape `[K=|Y|, V=|X|, S_u, S_u]`. K is the
            perception input alphabet; V is the action emission alphabet.
        t_initial_latent: initial belief over `S_t`. Defaults to uniform.
        u_initial_latent: initial belief over `S_u`. Defaults to uniform.
        u_initial_action: initial belief over the latched action `|X|`.
            Defaults to uniform.
        noise_epsilon: optional emission-noise epsilon for the joint process.
        device: jax device.

    Returns:
        FactoredGenerativeProcess with two factors (agent T at index 0,
        action-delayed agent U at index 1) and a `FeedbackTransducerStructure`.
    """
    if t_transducer.ndim != 4 or u_transducer.ndim != 4:
        raise ValueError("transducers must have shape [K, V, S, S]")
    K_t, V_t, S_t, _ = t_transducer.shape
    K_u, V_u, S_u, _ = u_transducer.shape
    if K_t != V_u:
        raise ValueError(
            f"agent T's input alphabet ({K_t}) must equal agent U's emission alphabet ({V_u})"
        )
    if K_u != V_t:
        raise ValueError(
            f"agent U's input alphabet ({K_u}) must equal agent T's emission alphabet ({V_t})"
        )

    action_size = V_u
    u_delayed = make_action_delayed(u_transducer, action_size=action_size)

    if t_initial_latent is None:
        t_initial_latent = jnp.ones(S_t) / S_t
    else:
        t_initial_latent = jnp.asarray(t_initial_latent, dtype=jnp.float32)
    if u_initial_latent is None:
        u_initial_latent = jnp.ones(S_u) / S_u
    else:
        u_initial_latent = jnp.asarray(u_initial_latent, dtype=jnp.float32)
    if u_initial_action is None:
        u_initial_action = jnp.ones(action_size) / action_size
    else:
        u_initial_action = jnp.asarray(u_initial_action, dtype=jnp.float32)

    u_initial = make_action_delayed_initial_state(u_initial_latent, u_initial_action)

    structure = FeedbackTransducerStructure(
        t_idx=0,
        u_indices=(1,),
        u_action_sizes=(action_size,),
        u_latent_sizes=(S_u,),
    )

    # The factor-1 transition matrix has K = V_t (perception input).
    # FactoredGenerativeProcess expects normalizing_eigenvectors[i] shape [K_i, S_i].
    # For HMM components we still pass a placeholder shaped correctly.
    t_norms = jnp.zeros((K_t, S_t))            # unused for "hmm" component
    u_norms = jnp.zeros((V_t, S_u * action_size))

    return FactoredGenerativeProcess(
        component_types=("hmm", "hmm"),
        transition_matrices=(t_transducer, u_delayed),
        normalizing_eigenvectors=(t_norms, u_norms),
        initial_states=(t_initial_latent, u_initial),
        structure=structure,
        noise_epsilon=noise_epsilon,
        device=device,
    )


def _instantiate_hmm(
    spec: Mapping[str, Any],
    device: str | None,
) -> jax.Array:
    """Look up an HMM kernel function by name and instantiate it. Returns [V, S, S].

    Only HMMs are accepted (GHMM kernels are not sum-normalizable into the
    transducer form this builder assumes)."""
    process_name = spec["process_name"]
    if process_name not in HMM_MATRIX_FUNCTIONS:
        raise KeyError(
            f'Unknown HMM process "{process_name}". '
            f"Available: {', '.join(sorted(HMM_MATRIX_FUNCTIONS))}"
        )
    fn = HMM_MATRIX_FUNCTIONS[process_name]
    params = dict(spec.get("process_params") or {})
    sig = inspect.signature(fn)
    try:
        with jax.default_device(resolve_jax_device(device)):
            sig.bind_partial(**params)
            M = fn(**params)
    except TypeError as e:
        sigstr = ", ".join(f"{k}: {v.annotation}" for k, v in sig.parameters.items())
        raise TypeError(
            f"Invalid params for HMM {process_name!r}: {e}. Signature: {sigstr}"
        ) from e
    if M.ndim != 3:
        raise ValueError(
            f"HMM {process_name!r} must return shape [V, S, S]; got {tuple(M.shape)}"
        )
    return M


def _build_transducer_tensor(
    spec: Mapping[str, Any],
    device: str | None,
) -> jax.Array:
    """Build a `[K, V, S, S]` transducer tensor from a spec.

    Two spec forms are supported:

    1. Registered transducer process:
           {process_name: <str>, process_params: {...}}
       Looked up in `TRANSDUCER_MATRIX_FUNCTIONS`; the function must return
       a 4-D `[K, V, S, S]` tensor.

    2. List of HMMs stacked along K:
           {from_hmms: [{process_name, process_params}, ...]}
       Each entry is looked up in `HMM_MATRIX_FUNCTIONS` and must return a
       3-D `[V, S, S]` tensor; all entries must share `(V, S)`. The result
       is stacked along a new leading K axis (K = len(from_hmms)): input k
       selects which HMM kernel runs.

    Exactly one of the two forms must be present.
    """
    has_proc = "process_name" in spec
    has_hmms = "from_hmms" in spec
    if has_proc == has_hmms:
        raise ValueError(
            "transducer spec must contain exactly one of 'process_name' or 'from_hmms'"
        )

    if has_hmms:
        hmm_specs = list(spec["from_hmms"] or [])
        if not hmm_specs:
            raise ValueError("'from_hmms' must contain at least one HMM spec")
        mats = [_instantiate_hmm(s, device) for s in hmm_specs]
        ref_shape = mats[0].shape
        for i, M in enumerate(mats[1:], start=1):
            if M.shape != ref_shape:
                raise ValueError(
                    f"from_hmms[{i}] has shape {tuple(M.shape)}, expected {tuple(ref_shape)} "
                    f"(all HMMs must share (V, S) so they can stack along K)."
                )
        with jax.default_device(resolve_jax_device(device)):
            return jnp.stack(mats, axis=0)  # [K=len(mats), V, S, S]

    process_name = spec["process_name"]
    if process_name not in TRANSDUCER_MATRIX_FUNCTIONS:
        raise KeyError(
            f'Unknown transducer process "{process_name}". '
            f"Available: {', '.join(sorted(TRANSDUCER_MATRIX_FUNCTIONS))}"
        )
    fn = TRANSDUCER_MATRIX_FUNCTIONS[process_name]
    params = dict(spec.get("process_params") or {})
    sig = inspect.signature(fn)
    try:
        with jax.default_device(resolve_jax_device(device)):
            sig.bind_partial(**params)
            T = fn(**params)
    except TypeError as e:
        sigstr = ", ".join(f"{k}: {v.annotation}" for k, v in sig.parameters.items())
        raise TypeError(f"Invalid params for transducer {process_name!r}: {e}. Signature: {sigstr}") from e
    if T.ndim != 4:
        raise ValueError(
            f"Transducer {process_name!r} must return shape [K, V, S, S]; got {tuple(T.shape)}"
        )
    return T


def build_feedback_transducer_process_from_spec(
    factors: Sequence[Mapping[str, Any]],
    noise_epsilon: float = 0.0,
    device: str | None = None,
) -> FactoredGenerativeProcess:
    """Hydra-targeted builder for feedback-transducer feedback processes.

    Spec format (all entries optional unless marked required):

        factors:
          - role: T                                # required: "T" exactly once
            transducer:                            # required (see forms below)
              process_name: <str>                  # registered transducer name
              process_params: {...}
            initial_latent: [...]                  # default: uniform over S
          - role: U                                # required for each U agent
            transducer:                            # required (see forms below)
              process_name: <str>
              process_params: {...}
            delay_action: true                     # default: true (action-delayed wrapper)
            initial_latent: [...]                  # default: uniform over S
            initial_action: [...]                  # default: uniform over |X_i|
                                                   # (used only when delay_action)

    A `transducer` block accepts either of two forms:

        # registered 4-D transducer
        transducer:
          process_name: <name in TRANSDUCER_MATRIX_FUNCTIONS>
          process_params: {...}

        # list of HMMs stacked along K (input k -> HMM kernel k)
        transducer:
          from_hmms:
            - {process_name: <name in HMM_MATRIX_FUNCTIONS>, process_params: {...}}
            - {process_name: ..., process_params: {...}}

    With `from_hmms`, all listed HMMs must share `(V, S)`; the resulting
    transducer has `K = len(from_hmms)`.

    Constraints:
        - exactly one factor with role="T"; one or more with role="U"
        - in v0, all factors must be HMM-style (sum-normalized). GHMM
          transducers are not supported yet.
        - agent T's K (input alphabet) must equal the radix-encoded vector
          of all U agents' emission alphabets, in declaration order.
        - each U agent's K (input alphabet) must equal agent T's emission
          alphabet V_t.

    Returns:
        FactoredGenerativeProcess with factor 0 = agent T and factors 1..N =
        action-delayed U agents (or raw U agents if delay_action=False), wired
        with `FeedbackTransducerStructure`.
    """
    if not factors:
        raise ValueError("factors must contain at least one entry")

    t_indices = [i for i, f in enumerate(factors) if f.get("role") == "T"]
    u_indices = [i for i, f in enumerate(factors) if f.get("role") == "U"]

    if len(t_indices) != 1:
        raise ValueError(f"factors must contain exactly one role='T'; got {len(t_indices)}")
    if not u_indices:
        raise ValueError("factors must contain at least one role='U'")
    other = [i for i, f in enumerate(factors) if f.get("role") not in ("T", "U")]
    if other:
        roles = [factors[i].get("role") for i in other]
        raise ValueError(f"unknown roles in factors: {roles}")

    t_idx = t_indices[0]

    # Build per-factor transducer tensors and metadata.
    transition_matrices: list[jax.Array] = [None] * len(factors)  # type: ignore[list-item]
    initial_states: list[jax.Array] = [None] * len(factors)       # type: ignore[list-item]
    component_types: list[str] = [None] * len(factors)            # type: ignore[list-item]
    norms_list: list[jax.Array] = [None] * len(factors)           # type: ignore[list-item]
    u_action_sizes: list[int] = []
    u_latent_sizes: list[int] = []

    # Agent T first (no delay)
    t_spec = factors[t_idx]
    t_tensor = _build_transducer_tensor(t_spec["transducer"], device)
    K_t, V_t, S_t, _ = t_tensor.shape
    transition_matrices[t_idx] = t_tensor
    component_types[t_idx] = "hmm"  # v0
    t_initial_latent = t_spec.get("initial_latent")
    initial_states[t_idx] = (
        jnp.asarray(t_initial_latent, dtype=jnp.float32)
        if t_initial_latent is not None
        else jnp.ones(S_t) / S_t
    )
    norms_list[t_idx] = jnp.zeros((K_t, S_t))  # unused for HMM

    # U agents in declaration order
    expected_u_K = V_t  # each U agent receives T's perception
    radix_check = 1
    for idx in u_indices:
        a_spec = factors[idx]
        delay = bool(a_spec.get("delay_action", True))
        U = _build_transducer_tensor(a_spec["transducer"], device)
        K_a, V_a, S_a, _ = U.shape
        if K_a != expected_u_K:
            raise ValueError(
                f"agent U at factors[{idx}] has K={K_a}, expected K={expected_u_K} "
                f"(agent T's V_t). Each U agent's input alphabet must match agent T's "
                f"emission alphabet."
            )
        u_action_sizes.append(int(V_a))
        u_latent_sizes.append(int(S_a))
        radix_check *= int(V_a)

        a_initial_latent = a_spec.get("initial_latent")
        a_latent_belief = (
            jnp.asarray(a_initial_latent, dtype=jnp.float32)
            if a_initial_latent is not None
            else jnp.ones(S_a) / S_a
        )

        if delay:
            U_d = make_action_delayed(U, action_size=V_a)
            a_initial_action = a_spec.get("initial_action")
            a_action_belief = (
                jnp.asarray(a_initial_action, dtype=jnp.float32)
                if a_initial_action is not None
                else jnp.ones(V_a) / V_a
            )
            transition_matrices[idx] = U_d
            initial_states[idx] = make_action_delayed_initial_state(a_latent_belief, a_action_belief)
            norms_list[idx] = jnp.zeros((K_a, S_a * V_a))
        else:
            # Without action-delay the structure's "self-determining" assumption
            # doesn't hold: this branch is currently a placeholder for non-delayed
            # agent-U factors and is not exercised by the structure in v0.
            raise NotImplementedError(
                f"factors[{idx}].delay_action=false is not supported yet; "
                f"the v0 FeedbackTransducerStructure assumes all U agents are "
                f"action-delayed (self-determining)."
            )

        component_types[idx] = "hmm"  # v0

    if radix_check != K_t:
        raise ValueError(
            f"agent T's K={K_t} but product of U agents' action alphabets is {radix_check}. "
            f"They must match: agent T's input alphabet = radix encoding of the U agents' emission tuple."
        )

    structure = FeedbackTransducerStructure(
        t_idx=t_idx,
        u_indices=tuple(u_indices),
        u_action_sizes=tuple(u_action_sizes),
        u_latent_sizes=tuple(u_latent_sizes),
    )

    return FactoredGenerativeProcess(
        component_types=tuple(component_types),
        transition_matrices=tuple(transition_matrices),
        normalizing_eigenvectors=tuple(norms_list),
        initial_states=tuple(initial_states),
        structure=structure,
        noise_epsilon=noise_epsilon,
        device=device,
    )


def lift_slot_projected(
    base: jax.Array,
    slot: int,
    u_action_sizes: Sequence[int],
) -> jax.Array:
    """Lift a single-input-slot transducer to the joint-action input space.

    Given `base` of shape `[A_slot, V, S, S]` whose K axis indexes only one
    agent's action, produce a `[K_t, V, S, S]` transducer where
    `K_t = prod(u_action_sizes)` and the K axis indexes the radix-
    encoded joint action tuple in the order used by
    `FeedbackTransducerStructure.select_variants` (first agent = most
    significant digit, last agent = least significant).

    The lifted kernel at joint index `k` equals `base` at the digit in
    position `slot` of `k`'s radix decomposition. All other digits of `k`
    are ignored.

    Args:
        base: `[A_slot, V, S, S]` transducer keyed on one agent's action.
        slot: position (0-indexed) of the agent whose action drives `base`.
        u_action_sizes: action alphabet sizes for all agents in their
            declaration order. `u_action_sizes[slot]` must equal
            `base.shape[0]`.

    Returns:
        `[K_t, V, S, S]` transducer.
    """
    if base.ndim != 4:
        raise ValueError(f"base must have shape [A_slot, V, S, S]; got {tuple(base.shape)}")
    n = len(u_action_sizes)
    if not 0 <= slot < n:
        raise ValueError(f"slot {slot} out of range for {n} agents")
    A_slot = int(u_action_sizes[slot])
    if base.shape[0] != A_slot:
        raise ValueError(
            f"base.shape[0]={base.shape[0]} must equal u_action_sizes[{slot}]={A_slot}"
        )
    K_t = math.prod(int(a) for a in u_action_sizes)
    # Radix weight at position `slot`: product of sizes strictly to the right.
    weight = math.prod(int(a) for a in u_action_sizes[slot + 1:])
    indices = jnp.arange(K_t, dtype=jnp.int32)
    digits = (indices // weight) % A_slot                                  # [K_t]
    return base[digits]                                                    # [K_t, V, S, S]


def build_telephone_loop(
    agents: Sequence[Mapping[str, Any]],
    action_size: int | None = None,
    noise_epsilon: float = 0.0,
    t_noise_epsilon: float = 0.0,
    device: str | None = None,
) -> FactoredGenerativeProcess:
    """Build a Telephone feedback-transducer process.

    N action-delayed U agents arranged in a cycle. Agent T is a
    memoryless deterministic router with `S_t = 1`: its K input is the
    radix-encoded joint action tuple, and it emits the same code as a single
    token in `[0, A^N)`. Each agent's lifted transducer reads the digit
    corresponding to its predecessor `(i - 1) mod N` from that token, so
    agent `i`'s percept at step `t` is agent `(i - 1) mod N`'s latched action
    from step `t - 1`.

    Args:
        agents: list of N>=2 agent specs. Each entry has a `transducer:` block
            in the same form as `build_feedback_transducer_process_from_spec`
            (registered `process_name` or `from_hmms`). The base transducer
            must have shape `[A, A, S_i, S_i]` (predecessor's action selects
            kernel; emission is in the same `A`-alphabet). May also include
            `initial_latent` and `initial_action`.
        action_size: action alphabet `A` shared by all agents. If `None`,
            inferred from the first agent's base transducer K.
        noise_epsilon: optional joint-output blur epsilon (applies to the
            full joint emission, including agent action symbols).
        t_noise_epsilon: probability that the router T emits a uniform
            random K-tuple instead of the deterministic copy of the latched
            agent actions. Affects only T's emission channel, not the
            U agents' own action emissions.
        device: jax device.

    Returns:
        FactoredGenerativeProcess with factor 0 = router T and factors
        1..N = action-delayed U agents in declaration order.
    """
    if not 0.0 <= float(t_noise_epsilon) <= 1.0:
        raise ValueError(
            f"t_noise_epsilon must be in [0, 1], got {t_noise_epsilon}"
        )
    if len(agents) < 2:
        raise ValueError(f"telephone requires at least 2 agents, got {len(agents)}")

    n_agents = len(agents)

    # Build per-agent base transducers and validate shapes.
    base_tensors: list[jax.Array] = []
    for i, a_spec in enumerate(agents):
        if "transducer" not in a_spec:
            raise ValueError(f"agents[{i}] missing 'transducer' block")
        U = _build_transducer_tensor(a_spec["transducer"], device)
        if U.ndim != 4:
            raise ValueError(
                f"agents[{i}] transducer must have shape [A, A, S, S]; got {tuple(U.shape)}"
            )
        K_a, V_a, S_a, S2 = U.shape
        if S_a != S2:
            raise ValueError(f"agents[{i}] transducer not square in S; got {tuple(U.shape)}")
        if K_a != V_a:
            raise ValueError(
                f"agents[{i}] base transducer must have K==V (predecessor alphabet == "
                f"self-emission alphabet); got K={K_a}, V={V_a}"
            )
        base_tensors.append(U)

    # Resolve / validate action_size.
    inferred_A = int(base_tensors[0].shape[0])
    if action_size is None:
        action_size = inferred_A
    elif int(action_size) != inferred_A:
        raise ValueError(
            f"action_size={action_size} disagrees with agents[0] base K={inferred_A}"
        )
    for i, U in enumerate(base_tensors):
        if U.shape[0] != action_size:
            raise ValueError(
                f"agents[{i}] base K={U.shape[0]} != action_size={action_size}; "
                f"telephone v0 requires a uniform action alphabet across agents"
            )

    u_action_sizes = tuple(int(action_size) for _ in range(n_agents))
    u_latent_sizes = tuple(int(U.shape[2]) for U in base_tensors)
    K_t = int(action_size) ** n_agents

    # Router T: copy(K_t), optionally with uniform-output noise. S_t = 1.
    with jax.default_device(resolve_jax_device(device)):
        t_tensor = _copy_transducer(alphabet_size=K_t)                      # [K_t, K_t, 1, 1]
        e = float(t_noise_epsilon)
        if e > 0.0:
            # P(v | k) = (1-e)·1{v=k} + e/K_t
            noise = jnp.full((K_t, K_t, 1, 1), e / K_t, dtype=t_tensor.dtype)
            t_tensor = (1.0 - e) * t_tensor + noise
    t_initial = jnp.ones(1, dtype=jnp.float32)
    t_norms = jnp.zeros((K_t, 1))                                       # unused for HMM

    # Lifted action-delayed U agents.
    transition_matrices: list[jax.Array] = [t_tensor]
    initial_states: list[jax.Array] = [t_initial]
    component_types: list[str] = ["hmm"]
    norms_list: list[jax.Array] = [t_norms]

    for i, U in enumerate(base_tensors):
        predecessor = (i - 1) % n_agents
        U_lifted = lift_slot_projected(U, slot=predecessor, u_action_sizes=u_action_sizes)
        U_d = make_action_delayed(U_lifted, action_size=int(action_size))

        a_spec = agents[i]
        S_a = int(U.shape[2])
        a_latent = a_spec.get("initial_latent")
        a_latent_belief = (
            jnp.asarray(a_latent, dtype=jnp.float32)
            if a_latent is not None
            else jnp.ones(S_a) / S_a
        )
        a_action = a_spec.get("initial_action")
        a_action_belief = (
            jnp.asarray(a_action, dtype=jnp.float32)
            if a_action is not None
            else jnp.ones(int(action_size)) / int(action_size)
        )
        transition_matrices.append(U_d)
        initial_states.append(make_action_delayed_initial_state(a_latent_belief, a_action_belief))
        component_types.append("hmm")
        norms_list.append(jnp.zeros((K_t, S_a * int(action_size))))

    structure = FeedbackTransducerStructure(
        t_idx=0,
        u_indices=tuple(range(1, n_agents + 1)),
        u_action_sizes=u_action_sizes,
        u_latent_sizes=u_latent_sizes,
    )

    return FactoredGenerativeProcess(
        component_types=tuple(component_types),
        transition_matrices=tuple(transition_matrices),
        normalizing_eigenvectors=tuple(norms_list),
        initial_states=tuple(initial_states),
        structure=structure,
        noise_epsilon=noise_epsilon,
        device=device,
    )


def _partition_transducer(
    perms: Sequence[Sequence[int]],
    partition: Sequence[int],
    eps: float,
    delta: float,
) -> jax.Array:
    """Transducer [K, V, S, S]: per input k, kernel = (1-eps)*P_perm + eps/S;
    emission = delta-noisy partition map of the ARRIVAL state (V = #classes)."""
    S = len(partition)
    V = int(max(partition)) + 1
    K = len(perms)
    T = np.zeros((K, V, S, S))
    emis = np.full((S, V), delta / V)
    emis[np.arange(S), np.asarray(partition)] += 1 - delta
    for k, pi in enumerate(perms):
        kernel = (1 - eps) * np.eye(S)[np.asarray(pi)] + eps / S
        for v in range(V):
            T[k, v] = kernel * emis[None, :, v]
    return jnp.asarray(T, dtype=jnp.float32)


def build_partition_loop(
    t_perms: Sequence[Sequence[int]],
    t_partition: Sequence[int],
    t_eps: float,
    t_delta: float,
    u_perms: Sequence[Sequence[int]],
    u_partition: Sequence[int],
    u_eps: float,
    u_delta: float,
    observe: str = "both",
    device: str | None = None,
):
    """Feedback loop of two partition-emission transducers, optionally
    coarse-grained to a single visible symbol.

    observe:
      "both" -> the FactoredGenerativeProcess over joint tokens (x, y).
      "x"    -> HMM over the joint latent observing ONLY the T symbol
                (agent U marginalized; complexification demonstrator).
      "y"    -> HMM observing only the U symbol.

    The coarse HMMs are exact marginalizations: the joint labeled operators
    are extracted from the factored process by delta-probing (from a product
    basis state, one filtering step decouples across factors), then grouped
    by the visible symbol. See apx:coarse_graining of the manuscript.
    """
    t_tensor = _partition_transducer(t_perms, t_partition, t_eps, t_delta)
    u_tensor = _partition_transducer(u_perms, u_partition, u_eps, u_delta)
    if observe == "both":
        return build_two_agent_loop(t_tensor, u_tensor, device=device)
    # Delta-probing must run at CPU float precision (GPU TF32 breaks the
    # stochasticity check); only the resulting coarse HMM lives on `device`.
    cpu_dev = jax.devices("cpu")[0]
    with jax.default_device(cpu_dev):
        probe_proc = build_two_agent_loop(t_tensor, u_tensor, device="cpu")
    return coarse_grain_two_agent_loop(probe_proc, observe,
                                       V_t=int(t_tensor.shape[1]), device=device)


def coarse_grain_two_agent_loop(process, observe: str, V_t: int,
                                device: str | None = None):
    """Exact marginalization of a two-agent feedback loop to a single-symbol HMM.

    `process` must be the FULLY OBSERVED loop, built on CPU (the delta-probing
    below is float-precision sensitive). `observe` selects which interface
    symbol stays visible:

      "x" -> only agent T's emission is observed (agent U marginalized out)
      "y" -> only agent U's action is observed  (agent T marginalized out)

    The joint labeled operators are extracted by delta-probing (from a product
    basis state one filtering step decouples across factors, so the operator
    row is P(v | i) * kron_f(posterior_f)), then summed over the hidden symbol.
    The result is an exact HMM over the joint latent: its beliefs are the
    coarse observer's Bayesian mixed states. See apx:coarse_graining.
    """
    dims = [int(np.asarray(s).shape[-1]) for s in process.initial_state]
    Z = int(np.prod(dims))
    V = int(process.vocab_size)
    grids = np.stack(np.meshgrid(*[np.arange(d) for d in dims], indexing="ij"),
                     -1).reshape(Z, len(dims))
    deltas = tuple(jnp.asarray(np.eye(d)[grids[:, f]], dtype=jnp.float32)
                   for f, d in enumerate(dims))
    P = np.asarray(jax.vmap(process.observation_probability_distribution)(deltas))
    trans = jax.vmap(process.transition_states)
    M = np.zeros((V, Z, Z))
    for v in range(V):
        obs = jnp.full((Z,), v, dtype=jnp.int32)
        post = [np.nan_to_num(np.asarray(q, dtype=np.float64))
                for q in trans(deltas, obs)]
        row = post[0]
        for q in post[1:]:
            row = (row[:, :, None] * q[:, None, :]).reshape(Z, -1)
        M[v] = np.where(P[:, v:v + 1] > 1e-12, P[:, v:v + 1] * row, 0.0)
    row_tot = M.sum(0).sum(1)
    if not np.allclose(row_tot, 1, atol=1e-3):
        raise RuntimeError(f"extracted joint operators are not stochastic "
                           f"(row sums in [{row_tot.min():.5f}, {row_tot.max():.5f}])")
    M = M / row_tot[None, :, None]

    V_u = V // V_t
    d0 = dims[0]
    probe = tuple(
        jnp.asarray(np.eye(d)[np.arange(d0) % d if f == 0 else np.zeros(d0, int)],
                    dtype=jnp.float32) for f, d in enumerate(dims))
    Pp = np.asarray(jax.vmap(process.observation_probability_distribution)(probe))
    ax_t = 0 if (Pp.reshape(-1, V_t, V_u).sum(2).std(0).max()
                   > Pp.reshape(-1, V_t, V_u).sum(1).std(0).max()) else 1
    tok = np.arange(V)
    x_id = (tok // V_u) if ax_t == 0 else (tok % V_u)
    y_id = (tok % V_u) if ax_t == 0 else (tok // V_u)
    group = x_id if observe == "x" else y_id
    n_sym = V_t if observe == "x" else V_u
    M_coarse = np.stack([M[group == sym].sum(0) for sym in range(n_sym)])

    b0 = np.asarray(process.initial_state[0], dtype=np.float64)
    for st in process.initial_state[1:]:
        b0 = np.kron(b0, np.asarray(st, dtype=np.float64))

    from fwh_core.generative_processes.hidden_markov_model import HiddenMarkovModel
    return HiddenMarkovModel(
        transition_matrices=jnp.asarray(M_coarse, dtype=jnp.float32),
        initial_state=jnp.asarray(b0, dtype=jnp.float32),
        device=device,
    )
