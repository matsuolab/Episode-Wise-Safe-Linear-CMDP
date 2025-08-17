from mpax import create_lp, r2HPDHG
from envs.cmdp import CMDP
import jax.numpy as jnp
import jax
from functools import partial
from itertools import product


# @partial(jax.jit, static_argnames=('solver'))
def compute_optimal_policy_LP(cmdp: CMDP, solver: r2HPDHG):
    H, S, A = cmdp.rew.shape
    B = jnp.zeros((H, S, A, H, S, A))

    cmdp.P.reshape()
    
    # Initial occupancy constraints
    B0 = jnp.eye(S)[:, None]
    B0 = jnp.repeat(B0, A, axis=1)
    B = B.at[0].set(B0)

    # Flow conservation constraints
    for h in range(1, H):
        for s, a in product(range(S), range(A)):
            B = B.at[h, s, a, h, s].set(1)
            B = B.at[h, s, a, h-1].set(-cmdp.P[h-1, :, :, s])

    # Initial occupancy constraints (h=0)
    B = B.at[0, :, :, 0, :].set(jnp.eye(S)[None, :, :])

    # Flow conservation constraints (h=1,...,H-1)
    # For each h, s, a, set B[h, s, a, h, s] = 1
    B = B.at[1:, :, :, 1:, :].set(jnp.eye(S)[None, :, :])
    # For each h, s, a, set B[h, s, a, h-1, :] = -cmdp.P[h-1, :, :, s]
    # cmdp.P shape: (H, S, A, S)
    # We want: B[h, s, a, h-1, :] = -cmdp.P[h-1, :, :, s]
    # So we need to broadcast over h, s, a
    idx_h = jnp.arange(1, H)
    B = B.at[idx_h[:, None, None], :, :, (idx_h - 1)[:, None, None], :].add(
        -cmdp.P[idx_h - 1].transpose(1, 2, 0)
    )



    B = B.reshape((H*S*A, H*S*A))
    mu = jnp.repeat(cmdp.init_dist[:, None], A, axis=1).reshape(-1)
    b = jnp.concatenate([mu, jnp.zeros((H-1)*S*A)])
    U = cmdp.utility.reshape(1, -1)
    u = jnp.array([cmdp.const])
    r = -cmdp.rew.reshape(-1)

    lp = create_lp(r, B, b, -U, -u, l=0, u=1.0)
    result = solver.optimize(lp)
    d_arr = result[0].reshape(H, S, A)

    nonneg_mask = d_arr >= 0
    d_arr = jnp.where(nonneg_mask, d_arr, 0.0)

    nonvisit_mask = d_arr.sum(axis=2, keepdims=True) == 0
    d_arr = jnp.where(nonvisit_mask, jnp.ones((H, S, A)) / A, d_arr)

    policy = d_arr / d_arr.sum(axis=-1, keepdims=True)
    return policy

