from jax.random import PRNGKey
import jax
import jax.numpy as jnp
from envs.cmdp import CMDP
import numpy as np


S, A = 10, 3
d = S * A
H = 5
K = 100000  # number of episodes
const_scale = 0.5


def create_cmdp(key: PRNGKey) -> CMDP:
    """Randomly generate synthetic CMDP instance."""
    const = 0  # dummy
    xi = 0  # dummy

    S_set = jnp.arange(S)
    A_set = jnp.arange(A)

    # feature map is one-hot vector
    d = S * A
    phi = jnp.eye(d)

    # create reward function
    rew = jnp.ones((H, S, A))
    key, _ = jax.random.split(key)
    zero_mask = jax.random.bernoulli(key, p=0.2, shape=rew.shape)
    rew = rew * zero_mask

    # create reward function for constraints
    utility = jnp.ones((H, S, A))
    key, _ = jax.random.split(key)
    zero_mask = jax.random.bernoulli(key, p=0.2, shape=utility.shape)
    utility = utility * zero_mask

    # create transition probability kernel
    key, _ = jax.random.split(key)
    P = jax.random.dirichlet(key=key, alpha=jnp.array([0.1] * S), shape=(H, S*A))
    P = P.reshape(H, S, A, S)

    # create initial distribution
    key, _ = jax.random.split(key)
    init_dist = jnp.zeros(S)
    x0 = jax.random.randint(key, (), 0, S)
    init_dist = init_dist.at[x0].set(1.0)

    cmdp = CMDP(S_set, A_set, H, d, phi, rew, utility, const, const_scale, P, init_dist, xi=xi)
    return cmdp



