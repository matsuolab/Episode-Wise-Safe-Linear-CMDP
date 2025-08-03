from scipy.optimize import linprog
from itertools import product
from functools import partial
from envs.cmdp import CMDP
from jax.random import PRNGKey
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
def compute_Q_h(Q_nh: jnp.ndarray, pol_nh: jnp.ndarray, bonus_h: jnp.ndarray, rew_h: jnp.ndarray, P_h: jnp.ndarray, ent_coef: float, thresh: float) -> jnp.ndarray:
    S, A = rew_h.shape
    Q_nh_reg = Q_nh + ent_coef * jax.scipy.special.entr(pol_nh)
    V_nh = (pol_nh * Q_nh_reg).sum(axis=1)
    next_v = bonus_h + P_h @ V_nh
    next_v = jnp.maximum(jnp.minimum(next_v, thresh), 0)
    chex.assert_shape(next_v, (S, A))
    return rew_h + next_v 


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
        thresh = thresh_coef * (1 + ent_coef * jnp.log(A)) * (H - h)

        Q_h = compute_Q_h(policy_Q[h+1], policy[h+1], jnp.zeros((S, A)), rew[h], P[h], ent_coef, thresh)
        policy_Q = policy_Q.at[h].set(Q_h)
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


def set_cmdp_info(cmdp: CMDP) -> CMDP:
    """Set CMDP information, such as optimal policy, constraint threshold, etc."""
    greedy_Q = compute_greedy_Q_utility(cmdp)
    safe_policy = compute_greedy_policy(greedy_Q)
    maximum_utility = (greedy_Q).max(axis=-1)[0] @ cmdp.init_dist
    const = maximum_utility * cmdp.const_scale
    xi = maximum_utility - const

    # Set safety related parameters
    cmdp = cmdp._replace(const=const, xi=xi, safe_policy=safe_policy)

    # set the optimal policy
    total_rew, _ = compute_optimal_rew_util(cmdp)
 
    cmdp = cmdp._replace(optimal_ret=total_rew)
    return cmdp


@jax.jit
def compute_optimal_rew_util(cmdp: CMDP, lr: float = 0.02, iter_length: int = 50000):
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


@jax.jit
def deploy_policy_episode(cmdp: CMDP, key: PRNGKey, policy: jnp.array):
    """ collect data through interaction to the cmdp 
    Args:
        cmdp (CMDP)
        H (int)
        key (PRNGKey)
        policy (jnp.ndarray)

    Returns:
        new_key (PRNGKey)
        traj (jnp.ndarray): (H x 3) collected trajectory H x (sa, s')
    """
    H, S, A, S = cmdp.P.shape
    chex.assert_shape(policy, (H, S, A))

    def body_fn(h, args):
        key, s, traj = args
        key, new_key = jax.random.split(key)
        act = jax.random.choice(new_key, A, p=policy[h, s])

        # sample next state
        key, new_key = jax.random.split(key)
        next_s = jax.random.choice(new_key, cmdp.S_set, p=cmdp.P[h, s, act])

        sa = s * A + act
        traj = traj.at[h].add(jnp.array([sa, next_s]))
        return key, next_s, traj

    key, init_key = jax.random.split(key)
    init_s = jax.random.choice(init_key, S, p=cmdp.init_dist)
    traj= jnp.zeros((H, 2), dtype=jnp.int32)  # H x (sa, s')
    args = key, init_s, traj
    key, _, traj = jax.lax.fori_loop(0, H, body_fn, args)
    return key, traj


@jax.jit
def sample_and_compute_regret(key, cmdp: CMDP, policy):
    """Deploy a policy and compute the regret"""
    # sample data and update visitation counter
    H, S, A, S = cmdp.P.shape

    key, init_key = jax.random.split(key)
    init_s = jax.random.choice(init_key, S, p=cmdp.init_dist)
    key, traj = deploy_policy_episode(cmdp, key, policy)

    # compute temporal regret
    Q_rew= EvalRegQ(policy, cmdp.rew, cmdp.P, 0)
    Q_utility = EvalRegQ(policy, cmdp.utility, cmdp.P, 0)
    init_dist = cmdp.init_dist
    total_rew = ((Q_rew * policy)[0].sum(axis=-1) * init_dist).sum()
    total_utility = ((Q_utility * policy)[0].sum(axis=-1) * init_dist).sum()
    
    err_rew = cmdp.optimal_ret - total_rew
    err_vio = jnp.maximum(cmdp.const - total_utility, 0)
    return key, traj, err_rew, err_vio


@jax.vmap
def Sherman_Morrison_update_H(Lambda_inv: jnp.ndarray, phi: jnp.ndarray) -> jnp.ndarray:
    """Update the inverse of the Lambda_matrix using the Sherman-Morrison formula.
    Due to the vmap, the input Lambda_inv is expected to be of shape (H x d x d) and phi of shape (H x d).
    
    Args:
        Lambda_inv (jnp.ndarray): (d x d) matrix
        phi (jnp.ndarray): d vector
    
    Return:
        Lambda_inv (jnp.ndarray): (d x d)
    """
    # Sherman-Morrison formula: (A + u v^T)^(-1) = A_inv - (A_inv u v^T A_inv) / (1 + v^T A_inv u)
    phi = phi.reshape(-1, 1)  # (d, 1)
    numerator = Lambda_inv @ phi @ phi.T @ Lambda_inv  # (d, d)
    denominator = 1.0 + (phi.T @ Lambda_inv @ phi)[0, 0]  # scalar
    return Lambda_inv - numerator / denominator