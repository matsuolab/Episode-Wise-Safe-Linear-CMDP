from mpax import create_lp, r2HPDHG
from mpax.utils import TerminationStatus
from envs.cmdp import CMDP
import jax.numpy as jnp
import jax
from jax.experimental.sparse import BCOO
from functools import partial


@jax.jit
def _flat_idx(h, s, a, sp, H, S, A):
    # flatten (h,s,a,s') in C-order
    return (((h * S + s) * A + a) * S + sp)


@jax.jit
def build_lp_jax_sparse(C, mu, P, bonus, utility, const, dtype=jnp.float32):
    C   = jnp.asarray(C,   dtype)
    mu  = jnp.asarray(mu,  dtype)
    P   = jnp.asarray(P,   dtype)
    bonus= jnp.asarray(bonus,dtype)
    utility = jnp.asarray(utility, dtype)

    H, S, A = C.shape
    n = H*S*A*S
    HS = H*S

    # objective & bounds
    c = jnp.broadcast_to(C[..., None], (H, S, A, S)).reshape(-1)

    # -------------------------
    # Build A (HS x n) as BCOO
    # rows for (h,s)
    h = jnp.arange(H)[:, None, None, None]       # (H,1,1,1)
    s = jnp.arange(S)[None, :, None, None]       # (1,S,1,1)
    a = jnp.arange(A)[None, None, :, None]       # (1,1,A,1)
    sp= jnp.arange(S)[None, None, None, :]       # (1,1,1,S)

    # Outflow: +1 * z[h,s,a,sp] in row r=(h,s)
    rows_out = jnp.broadcast_to(h*S + s, (H, S, A, S)).reshape(-1)
    cols_out = _flat_idx(h, s, a, sp, H, S, A).reshape(-1)
    vals_out = jnp.ones_like(cols_out, dtype)

    # Inflow for h>=1: -1 * z[h-1, s_prev, a_prev, s]
    h1  = jnp.arange(1, H)[:, None, None, None]
    hp  = h1 - 1
    sc  = jnp.arange(S)[None, :, None, None]
    spv = jnp.arange(S)[None, None, :, None]
    ap  = jnp.arange(A)[None, None, None, :]

    rows_in = jnp.broadcast_to(h1*S + sc, (H-1, S, S, A)).reshape(-1)
    cols_in = _flat_idx(hp, spv, ap, sc, H, S, A).reshape(-1)
    vals_in = -jnp.ones_like(cols_in, dtype)

    A_rows = jnp.concatenate([rows_out, rows_in])
    A_cols = jnp.concatenate([cols_out, cols_in])
    A_vals = jnp.concatenate([vals_out, vals_in])

    A_idx  = jnp.stack([A_rows, A_cols], axis=1)              # (nnz_A, 2)
    A_bcoo = BCOO((A_vals, A_idx), shape=(HS, n))

    b = jnp.concatenate([mu, jnp.zeros((H-1)*S, dtype)])

    # -------------------------
    # Build G (2n x n) as BCOO
    # For each i=(h,s,a,sp):
    #   Row i:        -1 on (i,i) and  +(P+β)[i] on all (h,s,a,y) for y=0..S-1
    #   Row i+n:      +1 on (i,i) and  -(P-β)[i] on all (h,s,a,y)
    idx_all = jnp.arange(n)
    # map flat i -> (h,s,a,sp)
    tmp1   = idx_all // S
    a_all  = tmp1 % A
    tmp2   = tmp1 // A
    s_all  = tmp2 % S
    h_all  = tmp2 // S

    # columns belonging to same (h,s,a, y)
    y = jnp.arange(S)                                      # (S,)
    cols_grp = _flat_idx(h_all[:, None], s_all[:, None], a_all[:, None], y[None, :], H, S, A)  # (n,S)

    Ppb = (P + bonus).reshape(-1)                           # length n
    Pmb = (P - bonus).reshape(-1)

    # Block 1 (rows 0..n-1)
    # diagonal -1
    rows1_d = idx_all
    cols1_d = idx_all
    vals1_d = -jnp.ones(n, dtype)
    # replicate (P+β) over y
    rows1_r = jnp.repeat(idx_all, S)
    cols1_r = cols_grp.reshape(-1)
    vals1_r = jnp.repeat(Ppb, S)

    # Block 2 (rows n..2n-1)
    rows2_d = idx_all + n
    cols2_d = idx_all
    vals2_d =  jnp.ones(n, dtype)
    rows2_r = jnp.repeat(idx_all + n, S)
    cols2_r = cols_grp.reshape(-1)
    vals2_r = -jnp.repeat(Pmb, S)

    G_rows = jnp.concatenate([rows1_d, rows1_r, rows2_d, rows2_r])
    G_cols = jnp.concatenate([cols1_d, cols1_r, cols2_d, cols2_r])
    G_vals = jnp.concatenate([vals1_d, vals1_r, vals2_d, vals2_r])

    # ---------- NEW row: (u_ext) z >= const ----------
    u_ext = jnp.broadcast_to(utility[..., None], (H, S, A, S)).reshape(-1)

    rows_u = jnp.full((n,), 2*n, dtype=jnp.int32)     # fixed-length row index
    cols_u = jnp.arange(n, dtype=jnp.int32)           # all columns
    vals_u = u_ext                                    # values (zeros OK)

    # concatenate into G
    G_rows = jnp.concatenate([G_rows, rows_u])
    G_cols = jnp.concatenate([G_cols, cols_u])
    G_vals = jnp.concatenate([G_vals, vals_u])

    G_idx  = jnp.stack([G_rows, G_cols], axis=1)           # (nnz_G, 2)
    G_bcoo = BCOO((G_vals, G_idx), shape=(2*n + 1, n))

    hvec = jnp.zeros(2*n + 1, dtype).at[-1].set(const)

    # c, l, u are dense vectors (as usual)
    return c, A_bcoo, b, G_bcoo, hvec


def compute_extended_optimal_policy(cmdp: CMDP, bonus: jnp.ndarray, safe_policy: jnp.ndarray):

    c, A, b, G, h = build_lp_jax_sparse(-cmdp.rew, cmdp.init_dist, cmdp.P, bonus, cmdp.utility, cmdp.const)

    lp = create_lp(c, A, b, G, h, 0.0, 1.0)                                  # sparse default
    solver = r2HPDHG(eps_abs=1e-3, eps_rel=1e-3, iteration_limit=10000)
    result = solver.optimize(lp)

    def policy_from_result():
        H, S, A = cmdp.rew.shape
        d_arr = jnp.asarray(result.primal_solution).reshape(H, S, A, S).sum(axis=-1)
        d_arr = jnp.maximum(d_arr, 0.0)
        d_arr = jnp.where(d_arr.sum(axis=-1, keepdims=True) > 0.0, d_arr, 1 / A)
        policy = d_arr / d_arr.sum(axis=-1, keepdims=True)
        return policy

    policy = jax.lax.cond(
        result.termination_status == TerminationStatus.OPTIMAL,
        policy_from_result,
        lambda: safe_policy,  # uniform policy if infeasible
    )
    return policy

@jax.jit
def compute_bonus_HSAS(count_HSAS):
    H, S, A, _ = count_HSAS.shape
    count_HSA = jnp.maximum(count_HSAS.sum(axis=-1, keepdims=True), 1)
    P_approx = jnp.where(count_HSAS.sum(axis=-1, keepdims=True) > 0, count_HSAS / count_HSA, 1 / S)
    Var = P_approx * (1 - P_approx)
    bonus = (jnp.sqrt(Var / count_HSA) + 1 / count_HSA)
    return bonus

