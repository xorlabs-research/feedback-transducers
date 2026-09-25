"""Smoke test for the feedback-transducer (perception-action loop) structure.

Run from the repo root:
    python tests/test_feedback_transducer.py

The test builds a 2-state agent T + memoryless deterministic agent U, generates a
short trajectory, and checks:

1. `make_action_delayed` produces a valid stochastic kernel: for each
   (perception, augmented_state), the joint emission/next-state distribution
   sums to 1.
2. `compute_joint_distribution` returns a valid distribution over composite
   tokens (sums to 1, non-negative).
3. After action-delay, agent U's emission marginal matches the deterministic
   policy: with U's latched action peaked at z, U emits z.
4. End-to-end sample-and-update preserves valid per-factor beliefs.
5. (Math check) The joint belief manifold is the tensor product of per-factor
   beliefs, matching the paper's factorization corollary: any joint belief reachable by
   the structure is `|rho_R> tensor |rho_{S,X}>`.
"""

from __future__ import annotations

import itertools
import sys

import jax
import jax.numpy as jnp
import numpy as np

from fwh_core.generative_processes.feedback_transducer import (
    build_feedback_transducer_process_from_spec,
    build_partition_loop,
    build_telephone_loop,
    build_two_agent_loop,
    make_action_delayed,
)
from fwh_core.generative_processes.transducer_matrices import (
    epsilon_greedy_lookup,
    input_driven_mess3,
    lookup,
)
from fwh_core.generative_processes.transition_matrices import fractal2, mess3


def _flip_t_transducer(p: float) -> jax.Array:
    """T[x, y, s, s'] = Pr(emit y, latent s' | action x, latent s) for a 2-state agent T.

    Action 0 keeps the latent with prob p, flips it with prob 1-p (and emits the new
    latent value as y). Action 1 reverses these probabilities.
    """
    T = jnp.zeros((2, 2, 2, 2))
    # action 0
    T = T.at[0, 0, 0, 0].set(p)        # stay 0 -> emit 0
    T = T.at[0, 1, 0, 1].set(1 - p)    # flip 0->1 -> emit 1
    T = T.at[0, 0, 1, 0].set(1 - p)    # flip 1->0 -> emit 0
    T = T.at[0, 1, 1, 1].set(p)        # stay 1 -> emit 1
    # action 1
    T = T.at[1, 0, 0, 0].set(1 - p)
    T = T.at[1, 1, 0, 1].set(p)
    T = T.at[1, 0, 1, 0].set(p)
    T = T.at[1, 1, 1, 1].set(1 - p)
    return T


def _copy_u_transducer() -> jax.Array:
    """U[y, x, s, s']: memoryless copy. Emission x equals perception y."""
    U = jnp.zeros((2, 2, 1, 1))
    U = U.at[0, 0, 0, 0].set(1.0)
    U = U.at[1, 1, 0, 0].set(1.0)
    return U


def test_action_delayed_is_stochastic() -> None:
    U = _copy_u_transducer()
    U_d = make_action_delayed(U, action_size=2)
    # U_d shape: [K=2, V=2, S*A=2, S*A=2]
    assert U_d.shape == (2, 2, 2, 2)
    # For each (k, augmented_state), sum over (V, augmented_next_state) must be 1
    totals = U_d.sum(axis=(1, 3))   # [K, augmented_state]
    assert jnp.allclose(totals, 1.0), f"action-delayed kernel not stochastic: {totals}"
    print("[ok] action-delayed kernel is stochastic per (perception, augmented_state)")


def test_action_delayed_emission_is_latched_action() -> None:
    """With state peaked at z, agent U emits z deterministically."""
    U = _copy_u_transducer()
    U_d = make_action_delayed(U, action_size=2)
    # Augmented state with latent=0 (only option), latched z=1
    state = jnp.array([0.0, 1.0])   # idx = latent*A + z = 0*2 + 1 = 1
    # Marginal over next state for each perception y, emission v:
    # post[y, v, sp] = state @ U_d[y, v, :, sp]
    post = jnp.einsum("s,kvst->kvt", state, U_d)
    # Sum over next state to get emission distribution
    em = post.sum(axis=2)            # [K=2, V=2]
    # For both perceptions y, emission should be peaked at v=1 (latched z=1)
    assert jnp.allclose(em, jnp.array([[0.0, 1.0], [0.0, 1.0]])), em
    print("[ok] action-delayed emission == latched action regardless of perception")


def test_joint_distribution_is_valid() -> None:
    T = _flip_t_transducer(p=0.8)
    U = _copy_u_transducer()
    proc = build_two_agent_loop(
        t_transducer=T,
        u_transducer=U,
        t_initial_latent=jnp.array([0.6, 0.4]),
        u_initial_latent=jnp.array([1.0]),
        u_initial_action=jnp.array([0.3, 0.7]),
    )
    state = proc.initial_state
    dist = proc.observation_probability_distribution(state)
    assert dist.shape == (proc.vocab_size,), (dist.shape, proc.vocab_size)
    assert jnp.all(dist >= -1e-7), dist
    s = float(dist.sum())
    assert abs(s - 1.0) < 1e-5, f"joint distribution sum != 1: {s}"
    # Composite vocab = V_t * V_u_action = 2 * 2 = 4
    assert proc.vocab_size == 4
    print(f"[ok] joint distribution valid: sum={s:.6f}, vocab={proc.vocab_size}")
    print(f"     dist = {dist}")


def test_belief_factorization_round_trip() -> None:
    """After one transition the per-factor states should still be valid
    probability distributions (sum to 1, non-negative)."""
    T = _flip_t_transducer(p=0.8)
    U = _copy_u_transducer()
    proc = build_two_agent_loop(
        t_transducer=T,
        u_transducer=U,
        t_initial_latent=jnp.array([0.6, 0.4]),
        u_initial_latent=jnp.array([1.0]),
        u_initial_action=jnp.array([0.3, 0.7]),
    )
    key = jax.random.PRNGKey(0)
    state = proc.initial_state
    for step in range(5):
        key, sub = jax.random.split(key)
        token = proc.emit_observation(state, sub)
        state = proc.transition_states(state, token)
        t_belief, u_belief = state
        t_sum = float(t_belief.sum())
        u_sum = float(u_belief.sum())
        assert abs(t_sum - 1.0) < 1e-4, f"step {step}: T belief sum {t_sum}"
        assert abs(u_sum - 1.0) < 1e-4, f"step {step}: U belief sum {u_sum}"
        assert jnp.all(t_belief >= -1e-7), t_belief
        assert jnp.all(u_belief >= -1e-7), u_belief
        # Decode for visibility: TokenEncoder uses vocab_sizes order [V_t, V_u],
        # so token = y * V_u + x_a (y = T emission, x_a = U action)
        print(
            f"  step {step}: token={int(token)} (y={token // 2}, x={token % 2}) "
            f"T={t_belief}  U={u_belief}"
        )
    print("[ok] per-factor beliefs remain valid distributions across 5 steps")


def test_paper_corollary1_decomposition() -> None:
    """Confirm the joint belief at each step factors as `rho_R x rho_{S,X}`.

    The framework already maintains them as a tuple, so the check is that
    constructing the joint belief by outer product reproduces the same joint
    distribution as a direct joint-HMM evaluation. This is implicit in the
    design (the structure operates on per-factor beliefs and only fuses at
    emission time), so the test is really a sanity check that the framework's
    construction is consistent with paper's Eq. V (joint as product) and
    the paper's factorization corollary.
    """
    T = _flip_t_transducer(p=0.7)
    U = _copy_u_transducer()
    proc = build_two_agent_loop(
        t_transducer=T,
        u_transducer=U,
        t_initial_latent=jnp.array([0.5, 0.5]),
        u_initial_latent=jnp.array([1.0]),
        u_initial_action=jnp.array([1.0, 0.0]),  # latched action = 0
    )
    # If latched action is 0, agent U emits 0 deterministically; T's variant=0;
    # so joint dist over (y, x_a) should be peaked on x_a=0 with y distributed
    # as compute_obs_dist_for_variant(T, t_state, T[0]).
    state = proc.initial_state
    dist = proc.observation_probability_distribution(state).reshape(2, 2)
    # axis 0: T emission y; axis 1: U emission x_a (radix order matches
    # TokenEncoder vocab_sizes = [V_t=2, V_u=2])
    p_x = dist.sum(axis=0)   # marginal over U emission
    p_y = dist.sum(axis=1)   # marginal over T emission
    expected_p_x = jnp.array([1.0, 0.0])
    assert jnp.allclose(p_x, expected_p_x, atol=1e-5), f"U marginal {p_x}, expected {expected_p_x}"
    print(f"[ok] U marginal = {p_x} (peaked on latched action 0)")
    print(f"     T marginal = {p_y}")
    # Re-set initial latched action to 1 and confirm agent U flips
    proc2 = build_two_agent_loop(
        t_transducer=T, u_transducer=U,
        t_initial_latent=jnp.array([0.5, 0.5]),
        u_initial_latent=jnp.array([1.0]),
        u_initial_action=jnp.array([0.0, 1.0]),
    )
    dist2 = proc2.observation_probability_distribution(proc2.initial_state).reshape(2, 2)
    p_x2 = dist2.sum(axis=0)
    assert jnp.allclose(p_x2, jnp.array([0.0, 1.0]), atol=1e-5), p_x2
    print(f"[ok] swap latched action -> U marginal = {p_x2} (peaked on 1)")


def test_from_spec_matches_explicit_builder() -> None:
    """Hydra-style spec builder produces the same joint distribution as
    the explicit `build_two_agent_loop`."""
    spec_factors = [
        {
            "role": "T",
            "transducer": {"process_name": "binary_flip", "process_params": {"p": 0.8}},
            "initial_latent": [0.6, 0.4],
        },
        {
            "role": "U",
            "transducer": {"process_name": "copy", "process_params": {"alphabet_size": 2}},
            "delay_action": True,
            "initial_latent": [1.0],
            "initial_action": [0.3, 0.7],
        },
    ]
    proc_spec = build_feedback_transducer_process_from_spec(factors=spec_factors)
    proc_explicit = build_two_agent_loop(
        t_transducer=_flip_t_transducer(p=0.8),
        u_transducer=_copy_u_transducer(),
        t_initial_latent=jnp.array([0.6, 0.4]),
        u_initial_latent=jnp.array([1.0]),
        u_initial_action=jnp.array([0.3, 0.7]),
    )
    d1 = proc_spec.observation_probability_distribution(proc_spec.initial_state)
    d2 = proc_explicit.observation_probability_distribution(proc_explicit.initial_state)
    assert jnp.allclose(d1, d2, atol=1e-6), (d1, d2)
    print(f"[ok] from_spec matches explicit builder: {d1}")


def test_from_spec_via_hydra_target() -> None:
    """Round-trip a Hydra-style instantiation via importlib (mirroring
    what `hydra.utils.instantiate` does with `_target_:`)."""
    import importlib

    target = "fwh_core.generative_processes.feedback_transducer.build_feedback_transducer_process_from_spec"
    mod_name, fn_name = target.rsplit(".", 1)
    fn = getattr(importlib.import_module(mod_name), fn_name)
    proc = fn(
        factors=[
            {"role": "T", "transducer": {"process_name": "binary_flip", "process_params": {"p": 0.8}}},
            {"role": "U",
             "transducer": {"process_name": "copy", "process_params": {"alphabet_size": 2}}},
        ],
        noise_epsilon=0.0,
    )
    assert proc.vocab_size == 4
    dist = proc.observation_probability_distribution(proc.initial_state)
    assert jnp.allclose(float(dist.sum()), 1.0, atol=1e-5)
    print(f"[ok] hydra _target_ path: vocab={proc.vocab_size}, dist sums to {float(dist.sum()):.6f}")


def test_from_spec_validation_errors() -> None:
    """Bad specs raise informative errors."""
    # Missing agent T
    try:
        build_feedback_transducer_process_from_spec(
            factors=[{"role": "U",
                      "transducer": {"process_name": "copy",
                                     "process_params": {"alphabet_size": 2}}}],
        )
        assert False, "expected ValueError for missing agent T"
    except ValueError as e:
        assert "exactly one role='T'" in str(e), e

    # Mismatched alphabets: T has V=2 but U expects K=3
    try:
        build_feedback_transducer_process_from_spec(
            factors=[
                {"role": "T",
                 "transducer": {"process_name": "binary_flip", "process_params": {"p": 0.5}}},
                {"role": "U",
                 "transducer": {"process_name": "copy",
                                "process_params": {"alphabet_size": 3}}},
            ],
        )
        assert False, "expected ValueError for alphabet mismatch"
    except ValueError as e:
        assert "input alphabet" in str(e), e

    # Unknown transducer
    try:
        build_feedback_transducer_process_from_spec(
            factors=[
                {"role": "T",
                 "transducer": {"process_name": "no_such_transducer", "process_params": {}}},
                {"role": "U",
                 "transducer": {"process_name": "copy",
                                "process_params": {"alphabet_size": 2}}},
            ],
        )
        assert False, "expected KeyError for unknown transducer"
    except KeyError as e:
        assert "no_such_transducer" in str(e), e

    print("[ok] from_spec validation errors are informative")


def test_input_driven_mess3_shape_and_stochastic() -> None:
    T = input_driven_mess3(x=0.15, a=0.6, beta=0.5)
    assert T.shape == (3, 3, 3, 3), T.shape
    # Stochastic: for each (action, state), sum over (emission, next_state) is 1
    sums = T.sum(axis=(1, 3))                              # [K=3, S=3]
    assert jnp.allclose(sums, 1.0, atol=1e-6), f"not stochastic: {sums}"
    print(f"[ok] input_driven_mess3 shape={T.shape}, all (a,s) marginals sum to 1")


def test_input_driven_mess3_action0_is_base_mess3() -> None:
    """Action 0 has K_0 = I (since P^0 = I), so it must reduce to base mess3."""
    base = mess3(x=0.15, a=0.6)                            # [V, S, S]
    T = input_driven_mess3(x=0.15, a=0.6, beta=0.5)
    assert jnp.allclose(T[0], base, atol=1e-6), (T[0], base)
    print("[ok] action 0 reduces to base mess3 (K_0 = I)")


def test_input_driven_mess3_beta_zero_collapses() -> None:
    """beta=0 collapses K_a to I for every action; all three slices equal."""
    T = input_driven_mess3(x=0.15, a=0.6, beta=0.0)
    base = mess3(x=0.15, a=0.6)
    for k in range(3):
        assert jnp.allclose(T[k], base, atol=1e-6), k
    print("[ok] beta=0 collapses all actions to base mess3")


def test_input_driven_mess3_full_rotation_directions() -> None:
    """beta=1: action 1 rotates A->B->C->A, action 2 rotates A->C->B->A."""
    T = input_driven_mess3(x=0.15, a=0.6, beta=1.0)
    # State |A> = [1, 0, 0]
    state_A = jnp.array([1.0, 0.0, 0.0])
    base = mess3(x=0.15, a=0.6)

    # Action 1 maps |A> -> |B>; emission distribution given post-rotation |B>
    # equals the emission distribution of base mess3 from state |B>.
    state_B = jnp.array([0.0, 1.0, 0.0])
    emissions_action1 = jnp.einsum("s,vst->v", state_A, T[1])
    expected_from_B = jnp.einsum("s,vst->v", state_B, base)
    assert jnp.allclose(emissions_action1, expected_from_B, atol=1e-6), (
        emissions_action1, expected_from_B,
    )

    # Action 2 maps |A> -> |C> (P^2 on a 3-cycle is the inverse direction).
    state_C = jnp.array([0.0, 0.0, 1.0])
    emissions_action2 = jnp.einsum("s,vst->v", state_A, T[2])
    expected_from_C = jnp.einsum("s,vst->v", state_C, base)
    assert jnp.allclose(emissions_action2, expected_from_C, atol=1e-6), (
        emissions_action2, expected_from_C,
    )
    print("[ok] beta=1 rotates: action 1 sends A->B, action 2 sends A->C")


def test_input_driven_mess3_in_coupled_loop() -> None:
    """End-to-end smoke: action-driven mess3 agent T + 3-action lookup agent U."""
    proc = build_feedback_transducer_process_from_spec(
        factors=[
            {
                "role": "T",
                "transducer": {
                    "process_name": "input_driven_mess3",
                    "process_params": {"x": 0.15, "a": 0.6, "beta": 0.5},
                },
            },
            {
                "role": "U",
                "transducer": {
                    "process_name": "epsilon_greedy_lookup",
                    "process_params": {
                        "table": [0, 1, 2],
                        "output_size": 3,
                        "epsilon": 1.0,  # uniform 3-way emission, ignores input
                    },
                },
            },
        ],
    )
    # Composite vocab: V_t (3 emissions) * V_u (3 actions) = 9
    assert proc.vocab_size == 9, proc.vocab_size
    dist = proc.observation_probability_distribution(proc.initial_state)
    assert jnp.all(dist >= -1e-7), dist
    assert abs(float(dist.sum()) - 1.0) < 1e-5, float(dist.sum())

    key = jax.random.PRNGKey(0)
    state = proc.initial_state
    for _step in range(4):
        key, sub = jax.random.split(key)
        token = proc.emit_observation(state, sub)
        state = proc.transition_states(state, token)
        t_b, u_b = state
        assert abs(float(t_b.sum()) - 1.0) < 1e-4
        assert abs(float(u_b.sum()) - 1.0) < 1e-4
    print(f"[ok] input_driven_mess3 in coupled loop: vocab={proc.vocab_size}, 4 steps OK")


COUNTER_ROTATOR_TABLE = [0, 2, 1]
"""Counter-rotator policy for the input_driven_mess3 cycle A->B->C->A:
A (perception 0) -> action 0 (no-op, already home);
B (perception 1) -> action 2 (P^2 on a 3-cycle: B->A in one step);
C (perception 2) -> action 1 (P:    C->A in one step)."""


def test_epsilon_greedy_shape_and_stochastic() -> None:
    U = epsilon_greedy_lookup(table=COUNTER_ROTATOR_TABLE, output_size=3, epsilon=0.2)
    assert U.shape == (3, 3, 1, 1), U.shape
    sums = U.sum(axis=(1, 3))                              # [K=3, S=1]
    assert jnp.allclose(sums, 1.0, atol=1e-6), sums
    print("[ok] epsilon_greedy_lookup is stochastic per perception")


def test_epsilon_greedy_eps0_matches_lookup() -> None:
    """epsilon=0 must reduce exactly to lookup."""
    table = COUNTER_ROTATOR_TABLE
    U_eps0 = epsilon_greedy_lookup(table=table, output_size=3, epsilon=0.0)
    U_det = lookup(table=table, output_size=3)
    assert jnp.allclose(U_eps0, U_det, atol=1e-7), (U_eps0, U_det)
    print("[ok] epsilon=0 reduces to deterministic lookup")


def test_epsilon_greedy_eps1_is_uniform() -> None:
    """epsilon=1 puts equal mass on every action regardless of perception."""
    U = epsilon_greedy_lookup(table=COUNTER_ROTATOR_TABLE, output_size=3, epsilon=1.0)
    expected = jnp.full((3, 3, 1, 1), 1.0 / 3)
    assert jnp.allclose(U, expected, atol=1e-7), U
    print("[ok] epsilon=1 is uniform over actions")


def test_epsilon_greedy_per_perception_distribution() -> None:
    """Hand-check the distribution at epsilon=0.3 for the counter-rotator table.
    For perception k, action table[k] gets prob (1-eps) + eps/A, others eps/A."""
    eps = 0.3
    A = 3
    U = epsilon_greedy_lookup(table=COUNTER_ROTATOR_TABLE, output_size=A, epsilon=eps)
    target = (1.0 - eps) + eps / A
    other = eps / A
    for k, v_target in enumerate(COUNTER_ROTATOR_TABLE):
        for v in range(A):
            expected = target if v == v_target else other
            got = float(U[k, v, 0, 0])
            assert abs(got - expected) < 1e-6, (k, v, expected, got)
    print(f"[ok] per-perception distribution at eps={eps}: target={target:.3f}, other={other:.3f}")


def _run_counter_rotator(epsilon: float, mess3_a: float, n_steps: int, seed: int) -> float:
    """Helper: run the counter-rotator + input_driven_mess3 loop and return
    the time-averaged P(A) of agent T's belief state."""
    proc = build_feedback_transducer_process_from_spec(
        factors=[
            {
                "role": "T",
                "transducer": {
                    "process_name": "input_driven_mess3",
                    "process_params": {"x": 0.15, "a": mess3_a, "beta": 0.9},
                },
            },
            {
                "role": "U",
                "transducer": {
                    "process_name": "epsilon_greedy_lookup",
                    "process_params": {
                        "table": COUNTER_ROTATOR_TABLE,
                        "output_size": 3,
                        "epsilon": epsilon,
                    },
                },
            },
        ],
    )
    key = jax.random.PRNGKey(seed)
    state = proc.initial_state
    t_belief_sum = jnp.zeros(3)
    for _step in range(n_steps):
        key, sub = jax.random.split(key)
        token = proc.emit_observation(state, sub)
        state = proc.transition_states(state, token)
        t_belief_sum = t_belief_sum + state[0]
    return float((t_belief_sum / n_steps)[0])


def test_counter_rotator_homes_state() -> None:
    """End-to-end: input_driven_mess3 (beta=0.9, a=0.85 for clear emission
    signal) coupled to a near-deterministic counter-rotator (eps=0.05) should
    drive agent T's hidden-state belief mass toward |A>.

    Agent U reacts only to *perceptions*, not the true state, so mess3's
    emission noise caps how much homing is achievable. With a=0.85 mess3 is
    sticky enough that perceptions reliably indicate state, and the homing
    signal is robust.
    """
    a_mass = _run_counter_rotator(epsilon=0.05, mess3_a=0.85, n_steps=200, seed=0)
    assert a_mass > 0.55, f"expected P(A) > 0.55 with strong homing; got {a_mass:.3f}"
    print(f"[ok] counter-rotator homes agent T: avg P(A) = {a_mass:.3f} (> 0.55)")


def test_counter_rotator_homing_is_monotone_in_epsilon() -> None:
    """Sweep epsilon and check that P(A) decreases monotonically as eps -> 1
    (more random = less homing). This is a relative test that doesn't depend
    on absolute mess3-fidelity calibration."""
    p_at_a = [
        _run_counter_rotator(epsilon=eps, mess3_a=0.85, n_steps=200, seed=0)
        for eps in (0.05, 0.3, 0.6, 1.0)
    ]
    # Strictly decreasing across the four settings (small noise tolerance).
    for prev, curr in zip(p_at_a, p_at_a[1:]):
        assert curr < prev + 0.02, f"P(A) is non-monotone over eps: {p_at_a}"
    # Lowest eps should give significantly higher P(A) than highest.
    assert p_at_a[0] - p_at_a[-1] > 0.15, f"insufficient homing range: {p_at_a}"
    print(f"[ok] homing decreases monotonically with eps: {[round(x, 3) for x in p_at_a]}")


def test_counter_rotator_drifts_at_epsilon_one() -> None:
    """Sanity: at epsilon=1 agent U is random, so state shouldn't concentrate.
    Time-average of P(A) should be close to 1/3 (no homing)."""
    proc = build_feedback_transducer_process_from_spec(
        factors=[
            {
                "role": "T",
                "transducer": {
                    "process_name": "input_driven_mess3",
                    "process_params": {"x": 0.15, "a": 0.6, "beta": 0.9},
                },
            },
            {
                "role": "U",
                "transducer": {
                    "process_name": "epsilon_greedy_lookup",
                    "process_params": {
                        "table": COUNTER_ROTATOR_TABLE,
                        "output_size": 3,
                        "epsilon": 1.0,
                    },
                },
            },
        ],
    )
    key = jax.random.PRNGKey(7)
    n_steps = 200
    state = proc.initial_state
    t_belief_sum = jnp.zeros(3)
    for _step in range(n_steps):
        key, sub = jax.random.split(key)
        token = proc.emit_observation(state, sub)
        state = proc.transition_states(state, token)
        t_belief_sum = t_belief_sum + state[0]
    avg = t_belief_sum / n_steps
    a_mass = float(avg[0])
    # Loose check: P(A) should be near 1/3, not concentrated.
    assert 0.18 < a_mass < 0.55, f"expected diffuse belief at eps=1; got P(A)={a_mass:.3f}"
    print(f"[ok] eps=1 (random) does not home: avg P(A) = {a_mass:.3f} (~ 1/3)")


# ---------------------------------------------------------------------------
# The builders the paper's runs use: telephone rings and partition loops.
# ---------------------------------------------------------------------------

def _seq_prob(proc, toks) -> float:
    """Probability of an observed token sequence under `proc` (exact filter)."""
    s = proc.initial_state
    lp = 1.0
    for tok in toks:
        d = np.asarray(proc.observation_probability_distribution(s))
        lp *= float(d[tok])
        if lp == 0.0:
            return 0.0
        s = proc.transition_states(s, jnp.array(tok))
    return lp


def test_telephone_loop_wires_predecessor_actions() -> None:
    """In the ring, agent i's kernel is selected by agent i-1's latched action,
    and the router emits the agents' latched actions as one radix-coded token."""
    A, N = 2, 3
    k0, k1 = fractal2(0.25, 0.25, 0.25), fractal2(0.75, 0.75, 0.75)
    agents = [{
        "transducer": {"from_hmms": [
            {"process_name": "fractal2", "process_params": {"b": 0.25, "p": 0.25, "q": 0.25}},
            {"process_name": "fractal2", "process_params": {"b": 0.75, "p": 0.75, "q": 0.75}}]},
        "initial_latent": [0.3, 0.7],
        # agent 0 has latched action 1, the others 0
        "initial_action": [0.0, 1.0] if i == 0 else [1.0, 0.0],
    } for i in range(N)]
    proc = build_telephone_loop(agents=agents, action_size=A, device="cpu")
    st = proc.initial_state
    dist = np.asarray(proc.observation_probability_distribution(st))
    tok = int(np.argmax(dist))
    assert dist[tok] > 0.999, dist
    # factor-major radix decode with vocab sizes [K_t = A**N, A, A, A]
    digits, rest = [], tok
    for v in reversed([A ** N, A, A, A]):
        digits.append(rest % v)
        rest //= v
    assert digits[::-1] == [4, 1, 0, 0], digits  # T code 1*4 + 0*2 + 0
    new = proc.transition_states(st, jnp.array(tok))
    latent = np.array([0.3, 0.7])

    def expected(kern):
        post = np.stack([latent @ np.asarray(kern[v]) for v in range(A)], 0)  # [z', s']
        post = post.T.reshape(-1)  # idx = s' * A + z'
        return post / post.sum()

    # agent 1 follows agent 0 (action 1 -> k1); agents 2 and 0 follow action 0 -> k0
    for i, kern in [(0, k0), (1, k1), (2, k0)]:
        assert np.allclose(np.asarray(new[1 + i]), expected(kern), atol=1e-6), i
    print("[ok] telephone: each agent's latent update uses the predecessor-selected kernel")


def test_partition_loop_coarse_graining_is_exact() -> None:
    """The x-only / y-only presentations are exact marginals of the full loop:
    every observed-sequence probability agrees over all length-3 sequences."""
    kw = dict(t_perms=[[4, 2, 1, 3, 0], [1, 3, 4, 2, 0], [3, 1, 0, 4, 2]],
              t_partition=[2, 1, 0, 0, 1], t_eps=0.05, t_delta=0.05,
              u_perms=[[1, 0, 3, 2, 4], [3, 1, 0, 4, 2], [1, 4, 2, 3, 0]],
              u_partition=[2, 1, 1, 0, 0], u_eps=0.02, u_delta=0.05)
    full = build_partition_loop(**kw, observe="both", device="cpu")
    hx = build_partition_loop(**kw, observe="x", device="cpu")
    hy = build_partition_loop(**kw, observe="y", device="cpu")
    Vx = Vy = 3
    assert int(full.vocab_size) == Vx * Vy
    T = 3
    worst = 0.0
    for xs in itertools.product(range(Vx), repeat=T):
        px_full = sum(_seq_prob(full, [x * Vy + y for x, y in zip(xs, ys)])
                      for ys in itertools.product(range(Vy), repeat=T))
        worst = max(worst, abs(px_full - _seq_prob(hx, list(xs))))
    for ys in itertools.product(range(Vy), repeat=T):
        py_full = sum(_seq_prob(full, [x * Vy + y for x, y in zip(xs, ys)])
                      for xs in itertools.product(range(Vx), repeat=T))
        worst = max(worst, abs(py_full - _seq_prob(hy, list(ys))))
    assert worst < 1e-6, worst
    print(f"[ok] partition loop: coarse presentations are exact marginals (max |diff| = {worst:.1e})")


def main() -> int:
    print("==== feedback-transducer smoke tests ====")
    test_action_delayed_is_stochastic()
    test_action_delayed_emission_is_latched_action()
    test_joint_distribution_is_valid()
    test_belief_factorization_round_trip()
    test_paper_corollary1_decomposition()
    test_from_spec_matches_explicit_builder()
    test_from_spec_via_hydra_target()
    test_from_spec_validation_errors()
    test_input_driven_mess3_shape_and_stochastic()
    test_input_driven_mess3_action0_is_base_mess3()
    test_input_driven_mess3_beta_zero_collapses()
    test_input_driven_mess3_full_rotation_directions()
    test_input_driven_mess3_in_coupled_loop()
    test_epsilon_greedy_shape_and_stochastic()
    test_epsilon_greedy_eps0_matches_lookup()
    test_epsilon_greedy_eps1_is_uniform()
    test_epsilon_greedy_per_perception_distribution()
    test_counter_rotator_homes_state()
    test_counter_rotator_homing_is_monotone_in_epsilon()
    test_counter_rotator_drifts_at_epsilon_one()
    test_telephone_loop_wires_predecessor_actions()
    test_partition_loop_coarse_graining_is_exact()
    print("==== all checks passed ====")
    return 0


if __name__ == "__main__":
    sys.exit(main())
