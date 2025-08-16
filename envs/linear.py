from jax.random import PRNGKey
import jax
import jax.numpy as jnp
from envs.cmdp import CMDP
import numpy as np


def create_cmdp(key: PRNGKey, S: int=5, A: int=3, d: int=2, H: int=8, const_scale: float=0.6) -> CMDP:
    """Randomly generate synthetic CMDP instance."""
    const = 0  # dummy
    xi = 0  # dummy

    S_set = jnp.arange(S)
    A_set = jnp.arange(A)

    # create a linear CMDP based on https://arxiv.org/pdf/2106.06239#page=10.27
    phi = jax.random.dirichlet(key=key, alpha=jnp.array([0.1] * d), shape=(S*A))
    phi = phi.reshape(S*A, d)
    key, _ = jax.random.split(key)

    mu = jax.random.dirichlet(key=key, alpha=jnp.array([0.1] * S), shape=(H*d))
    mu = mu.reshape(H, d, S)
    key, _ = jax.random.split(key)

    P = jnp.einsum('hds,kd->hks', mu, phi).reshape(H, S, A, S)
    np.testing.assert_allclose(P.sum(axis=-1), 1, atol=1e-5)

    # create reward function
    theta_r = jax.random.uniform(key=key, shape=(H, d))
    key, _ = jax.random.split(key)
    rew = jnp.einsum('hd,kd->hk', theta_r, phi).reshape(H, S, A)

    # create reward function for constraints
    theta_u = jax.random.uniform(key, shape=(H, d))
    key, _ = jax.random.split(key)
    utility = jnp.einsum('hd,kd->hk', theta_u, phi).reshape(H, S, A)

    # create initial distribution
    key, _ = jax.random.split(key)
    init_dist = jnp.zeros(S)
    x0 = jax.random.randint(key, (), 0, S)
    init_dist = init_dist.at[x0].set(1.0)
    np.testing.assert_allclose(init_dist.sum(axis=-1), 1, atol=1e-6)

    cmdp = CMDP(S_set, A_set, H, d, phi, rew, utility, const, const_scale, P, init_dist, xi=xi)
    return cmdp