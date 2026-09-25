"""Feedback-transducer (perception-action loop) structure.

Implements the joint feedback HMM `V^(xy)_{rsz -> r's'z'} = U^(z'|y)_{s,s'}
T^(y|z)_{r,r'} delta_{z,x}` from "Beliefs in feedback: Factored
representations of interacting agents in transformers", organized so that
per-factor beliefs stay factored across steps (the paper's factorization
corollary: belief states remain in `R x (S x X)`).

Convention for v0
-----------------
- Factor 0 is agent T (driven): its `K` axis is the joint encoding of all
  U agents' action emissions. Its tensor has shape `[prod(V_a_i), V_t, S_t, S_t]`.
- Factors 1..N are the action-delayed U agents (self-determining): emission is
  the previously-latched action, deterministically read from state. Their
  tensors have shape `[V_t, V_a_i, S_a_i*V_a_i, S_a_i*V_a_i]` after
  `make_action_delayed`. Variant axis indexes agent T's perception emission.

Augmented U-agent state convention: `idx = latent_idx * action_size + latched_idx`
(action index inner). Use `make_action_delayed` in `feedback_transducer.py` to construct
the tensor; use `make_action_delayed_initial_state` for the initial belief.
"""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp

from fwh_core.generative_processes.structures.protocol import ConditionalContext
from fwh_core.utils.factoring_utils import compute_obs_dist_for_variant


class FeedbackTransducerStructure(eqx.Module):
    """One agent-T factor (driven) + N action-delayed agent-U factors (self-determining).

    Attributes:
        t_idx: index of the agent-T factor in the factor tuple.
        u_indices: indices of the action-delayed agent-U factors, in the
            order their emissions are radix-encoded into agent T's variant axis.
        u_action_sizes: action-alphabet size for each agent-U factor (in
            u_indices order). Equals each U agent's emission vocab size and
            the inner-axis size of its augmented state.
        u_latent_sizes: latent-state size for each agent-U factor (S_i, not S_i*A_i).
    """

    t_idx: int
    u_indices: tuple[int, ...]
    u_action_sizes: tuple[int, ...]
    u_latent_sizes: tuple[int, ...]

    def compute_joint_distribution(self, context: ConditionalContext) -> jax.Array:
        """Joint emission distribution P(y, x_1, ..., x_N | t_state, u_states).

        For action-delayed U agents the emission is the previously-latched action,
        deterministic given state. Agent T's emission then mixes over the U agents'
        emission distributions via its `K` (variant) axis.
        """
        # 1. Per-U-agent emission distributions (marginal over latent of the augmented state)
        u_dists: list[jax.Array] = []
        for i, idx in enumerate(self.u_indices):
            S_i = self.u_latent_sizes[i]
            A_i = self.u_action_sizes[i]
            u_dists.append(context.states[idx].reshape(S_i, A_i).sum(axis=0))

        # 2. Joint distribution over U agents' emissions (independent given states)
        if len(u_dists) == 1:
            p_u = u_dists[0]                      # [A_0]
        else:
            p_u = u_dists[0]
            for d in u_dists[1:]:
                p_u = jnp.tensordot(p_u, d, axes=0)  # outer product
            # shape: [A_0, A_1, ..., A_{N-1}]
        K_t = math.prod(self.u_action_sizes)
        p_u_flat = p_u.reshape(K_t)

        # 3. Agent T's emission distribution under each variant (= each joint action)
        t_tensor = context.transition_matrices[self.t_idx]
        t_state = context.states[self.t_idx]
        t_ct = context.component_types[self.t_idx]
        t_norms = context.normalizing_eigenvectors[self.t_idx]

        def t_dist_under_k(k: jax.Array) -> jax.Array:
            T_k = t_tensor[k]
            norm_k = t_norms[k] if t_ct == "ghmm" else None
            return compute_obs_dist_for_variant(t_ct, t_state, T_k, norm_k)

        all_t = jax.vmap(t_dist_under_k)(jnp.arange(K_t))  # [K_t, V_y]
        V_y = t_tensor.shape[1]

        # 4. Joint over (T emission, U emissions)
        # joint[y, k] = all_t[k, y] * p_u_flat[k]; reshape K -> per-U-agent shape;
        # axis order in the framework is by factor index, so we need the result in
        # the framework's expected radix order: vocab_sizes[0..F-1]
        joint_yk = all_t.T * p_u_flat[None, :]   # [V_y, K_t]
        joint = joint_yk.reshape((V_y,) + tuple(self.u_action_sizes))
        # joint axes are now [F_t, F_u_0, F_u_1, ...] in u_indices order.
        # Permute to match the framework's factor ordering (factor index i has axis i).
        num_factors = len(context.vocab_sizes)
        # current_axis[factor_idx] = current position of that factor's axis in `joint`
        current_axis = {self.t_idx: 0}
        for i, idx in enumerate(self.u_indices):
            current_axis[idx] = 1 + i
        perm = tuple(current_axis[i] for i in range(num_factors))
        joint = jnp.transpose(joint, perm)
        return joint.reshape(-1)

    def select_variants(
        self,
        obs_tuple: tuple[jax.Array, ...],
        context: ConditionalContext,  # pylint: disable=unused-argument
    ) -> tuple[jax.Array, ...]:
        """Variant indices for the post-emission state update.

        - Agent T (driven): variant = radix-encoded vector of U agents' emissions.
        - Each action-delayed U agent: variant = agent T's emitted perception.
        """
        # Radix-encode U agents' emissions into agent T's K index.
        t_variant = jnp.array(0, dtype=jnp.int32)
        radix = 1
        for i in range(len(self.u_indices) - 1, -1, -1):
            idx = self.u_indices[i]
            A_i = self.u_action_sizes[i]
            t_variant = t_variant + obs_tuple[idx].astype(jnp.int32) * radix
            radix *= A_i

        t_emission = obs_tuple[self.t_idx].astype(jnp.int32)

        num_factors = len(obs_tuple)
        out: list[jax.Array] = [jnp.array(0, dtype=jnp.int32)] * num_factors
        out[self.t_idx] = t_variant
        for idx in self.u_indices:
            out[idx] = t_emission
        return tuple(out)

    def get_required_params(self) -> dict[str, type]:
        return {
            "t_idx": int,
            "u_indices": tuple,
            "u_action_sizes": tuple,
            "u_latent_sizes": tuple,
        }
