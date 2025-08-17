from mpax import create_lp, r2HPDHG
from envs.cmdp import CMDP
import jax.numpy as jnp
import jax
from functools import partial
from itertools import product


# @partial(jax.jit, static_argnames=('solver'))
def compute_extended_optimal_policy(cmdp: CMDP, bonus: jnp.ndarray, solver: r2HPDHG):
    H, S, A = cmdp.rew.shape
    B = jnp.zeros((H, S, A, S, H, S, A, S))

    # Initial occupancy constraints
    B0 = jnp.eye(S)[:, :, None]  # (S, S, 1)
    B0 = jnp.repeat(B0, A**2, axis=-1).reshape(S, S, A, A)
    B0 = jnp.swapaxes(B0, 1, 2)

    B = B.at[0, :, :, 0, :, :].set(B0)

    # =====
    Bh_h_sum = jnp.eye((H-1)*S)[:, :, None]  # (H-1*S, H-1*S, 1)
    Bh_h_sum = jnp.repeat(Bh_h_sum, A**2, axis=-1).reshape((H-1), S, (H-1), S, A, A)
    Bh_h_sum = jnp.transpose(Bh_h_sum, axes=(0, 1, 4, 2, 3, 5))
    B = B.at[1:, :, :, 1:, :, :].set(Bh_h_sum)

    # indices for h and (h-1)
    idx = jnp.arange(1, H)                       # shape (H-1,)
    rhs = -cmdp.P[:-1].transpose(0, 3, 1, 2)         # (H-1, S, S, A)
    rhs = jnp.broadcast_to(rhs[:, :, None, :, :], (H-1, S, A, S, A))  # (H-1, S, A, S, A)
    B = B.at[idx, :, :, idx - 1, :, :].set(rhs)
    # =====

    B = B.reshape((H*S*A, H*S*A))
    mu = jnp.repeat(cmdp.init_dist[:, None], A, axis=1).reshape(-1)
    b = jnp.concatenate([mu, jnp.zeros((H-1)*S*A)])
    U = cmdp.utility.reshape(1, -1)
    u = jnp.array([cmdp.const])
    r = -cmdp.rew.reshape(-1)

    lp = create_lp(r, B, b, U, u, l=0, u=1.0)
    result = solver.optimize(lp)
    d_arr = result[0].reshape(H, S, A)
    d_arr = jnp.maximum(d_arr, 0.0)

    nonvisit_mask = d_arr.sum(axis=2, keepdims=True) == 0
    d_arr = jnp.where(nonvisit_mask, jnp.ones((H, S, A)) / A, d_arr)

    policy = d_arr / d_arr.sum(axis=-1, keepdims=True)
    return policy

