import jax
import jax.numpy as jnp
from envs.cmdp import CMDP
from utils import compute_Q_h


@jax.jit
def compute_softmax_pol_h_Ghosh(rQ_h, uQ_h, ent_coef, lam):
    Q_h = rQ_h + lam * uQ_h
    return jax.nn.softmax(Q_h / ent_coef, axis=-1)


@jax.jit
def compute_softmax_pol_Ghosh(bonus: jnp.ndarray, cmdp: CMDP, ent_coef: float, lam: float, Cr: float, Cu: float) -> jnp.ndarray:
    H, S, A = bonus.shape

    def backup(i, args):
        rQ, uQ, pol = args
        h = H - i - 1

        thresh = H - h

        rQ_h = compute_Q_h(rQ[h+1], pol[h+1], Cr * bonus[h], cmdp.rew[h], cmdp.P[h], 0, thresh)

        uQ_h = compute_Q_h(uQ[h+1], pol[h+1], Cu * bonus[h], cmdp.utility[h], cmdp.P[h], 0, thresh)

        pol_h = compute_softmax_pol_h_Ghosh(rQ_h, uQ_h, ent_coef, lam)

        rQ = rQ.at[h].set(rQ_h)
        uQ = uQ.at[h].set(uQ_h)
        pol = pol.at[h].set(pol_h)
        return rQ, uQ, pol

    rQ = jnp.zeros((H+1, S, A))
    uQ = jnp.zeros((H+1, S, A))
    pol = jnp.ones((H+1, S, A)) / A

    args = rQ, uQ, pol
    rQ, uQ, pol = jax.lax.fori_loop(0, H, backup, args)

    uV = jnp.sum(uQ * pol, axis=-1)
    total_util = cmdp.init_dist @ uV[0]
    return total_util, pol[:-1]


@jax.jit
def bisection_search_lam_Ghosh(cmdp: CMDP, bonus: jnp.ndarray, ent_coef: float, Cr: float, Cu: float, Clam: float, iter_length: int) -> jnp.ndarray:
 
    def body_fn(_, lam_range):
        lam_low, lam_high = lam_range
        mid_lam = (lam_low + lam_high) / 2
        util, _ = compute_softmax_pol_Ghosh(bonus, cmdp, ent_coef, mid_lam, Cr, Cu)

        is_util_safe = util >= cmdp.const
        next_lam_range = jax.lax.cond(
            is_util_safe,
            lambda: jnp.array((lam_low, mid_lam)),
            lambda: jnp.array((mid_lam, lam_high))
        )
        return next_lam_range

    lam_range = jax.lax.fori_loop(0, iter_length, body_fn, jnp.array((0.0, Clam), dtype=jnp.float32))
    return lam_range

 