from scipy.optimize import linprog
from itertools import product
from functools import partial
from envs.cmdp import CMDP
import jax
import jax.numpy as jnp
import chex


@jax.jit
def compute_greedy_Q_utility(cmdp: CMDP):
    """
    Compute a greedy Q function with respect to the utility function
    Useful for setting the constraint threshold
    """
    H, S, A = cmdp.rew.shape
    def backup(i, greedy_Q):
        h = H - i - 1
        V = greedy_Q[h+1].max(axis=1)
        next_v = cmdp.P[h] @ V
        chex.assert_shape(next_v, (S, A))
        greedy_Q = greedy_Q.at[h].set(cmdp.utility[h] + next_v)
        return greedy_Q

    greedy_Q = jnp.zeros((H+1, S, A))
    greedy_Q = jax.lax.fori_loop(0, H, backup, greedy_Q)
    return greedy_Q[:-1]


@jax.jit
def EvalRegQ(policy: jnp.ndarray, rew: jnp.ndarray, P: jnp.ndarray, ent_coef: float, thresh_coef: float = 1.0) -> jnp.ndarray:
    """ Compute value function
    Args:
        policy (np.ndarray): (HxSxA) array
        rew (np.ndarray): (HxSxA) array
        P (np.ndarray): (HxSxSxA) array
        ent_coef (float): regularization parameter
        thresh_coef (float): threshold parameter

    Returns:
        Q (jnp.ndarray): (HxSxA)
    """

    H, S, A = policy.shape
    def backup(i, args):
        policy_Q= args
        h = H - i - 1
        Q = policy_Q[h+1] + ent_coef * jax.scipy.special.entr(policy[h+1])
        V = (policy[h+1] * Q).sum(axis=1)
        next_v = P[h] @ V
        chex.assert_shape(next_v, (S, A))
        policy_Q = policy_Q.at[h].set(rew[h] + next_v)
        min_thresh = thresh_coef * (1 + ent_coef * jnp.log(A)) * (H - h)
        policy_Q = jnp.maximum(jnp.minimum(policy_Q, min_thresh), 0)
        return policy_Q

    policy_Q= jnp.zeros((H+1, S, A))
    args = policy_Q
    policy_Q= jax.lax.fori_loop(0, H, backup, args)
    return policy_Q[:-1]


@jax.jit
def compute_greedy_policy(Q: jnp.ndarray):
    """ Compute a greedy policy with respect to the Q function"""
    greedy_policy = jnp.zeros_like(Q)
    H, S, A = Q.shape
    
    def body_fn(i, greedy_policy):
        greedy_policy = greedy_policy.at[i, jnp.arange(S), Q[i].argmax(axis=-1)].set(1)
        return greedy_policy

    greedy_policy = jax.lax.fori_loop(0, H, body_fn, greedy_policy)
    chex.assert_shape(greedy_policy, (H, S, A))
    return greedy_policy



@jax.jit
def compute_policy_matrix(policy: jnp.ndarray):
    """
    Args:
        policy (jnp.ndarray): (HxSxA) matrix

    Returns:
        policy_matrix (jnp.ndarray): (HxSxSA) matrix
    """
    H, S, A = policy.shape
    PI = policy.reshape(H, 1, S, A)
    PI = jnp.tile(PI, (1, S, 1, 1))
    eyes = jnp.tile(jnp.eye(S).reshape(1, S, S, 1), (H, 1, 1, 1))
    PI = (eyes * PI).reshape(H, S, S*A)
    return PI


@jax.jit
def compute_occupancy_measure(cmdp: CMDP, policy: jnp.ndarray):
    """
    Args:
        cmdp (CMDP)
        policy (jnp.ndarray): (HxSxA) matrix

    Returns:
        occ (jnp.ndarray): (HxSxA) matrix
    """
    H, S, A = policy.shape
    Pi = compute_policy_matrix(policy)
    P = cmdp.P.reshape(H, S*A, S)

    def body_fn(h, occ):
        next_occ = occ[h] @ P[h] @ Pi[h+1]
        occ = occ.at[h+1].set(next_occ)
        return occ
    
    occ = jnp.zeros((H+1, S*A))
    occ = occ.at[0].set((cmdp.init_dist @ Pi[0]))
    occ = jax.lax.fori_loop(0, cmdp.H, body_fn, occ)
    occ = occ[:-1].reshape(H, S, A)
    return occ


@jax.jit
def compute_optimal_rew_util(cmdp: CMDP, lr: float = 0.05, iter_length: int = 500):
    # Compute the optimal policy using Lagrange method
    H, S, A = cmdp.rew.shape

    def body_fn(i, args):
        avg_occ, lam = args

        ru = cmdp.rew + lam * cmdp.utility
        pol = compute_greedy_policy(compute_greedy_Q_utility(cmdp._replace(utility=ru)))
        occ = compute_occupancy_measure(cmdp, pol)
        avg_occ = avg_occ + occ / iter_length
        total_utility = (occ * cmdp.utility).sum()
        lam = jnp.maximum(lam - lr * (total_utility - cmdp.const), 0)
        return avg_occ, lam

    lam = 0.0
    avg_occ = jnp.zeros((H, S, A))

    avg_occ, _ = jax.lax.fori_loop(0, iter_length, body_fn, (avg_occ, lam))
    optimal_rew = (avg_occ * cmdp.rew).sum()
    optimal_util = (avg_occ * cmdp.utility).sum()
    return optimal_rew, optimal_util
