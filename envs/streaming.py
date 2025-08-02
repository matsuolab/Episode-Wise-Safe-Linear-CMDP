from jax.random import PRNGKey
import jax
import jax.numpy as jnp
from envs.cmdp import CMDP
import numpy as np


buff = 20  # buffer size
S, A = buff+1, 2  # state and action space sizes
d = S * A
H = 10 
K = 500000  # number of episodes
const_scale = 0.6


def create_cmdp(key: PRNGKey) -> CMDP:
    """Randomly generate media streaming CMDP instance."""
    const = 0  # dummy
    xi = 0  # dummy

    S_set = jnp.arange(S)
    A_set = jnp.arange(A)

    # feature map is one-hot vector
    d = S * A
    phi = jnp.eye(d)

    # create reward function
    rew = jnp.zeros((H, S, A))
    rew = rew.at[:, int(buff * 0.3):, :].set(1)  # +1 reward when the buffer is sufficiently full
    assert rew.shape == (H, S, A)

    # create reward function for constraints
    utility = jnp.zeros((H, S, A))
    utility = utility.at[:, :, 1].set(1)  # +1 reward when taking a slow transmission
    assert utility.shape == (H, S, A)

    # create transition probability kernel
    key, _ = jax.random.split(key)

    mu1 = jax.random.uniform(key, minval=0.5, maxval=0.9)  # fast transmission rate
    mu2 = 1 - mu1  # late transmission rate
    key, _ = jax.random.split(key)
    rho = jax.random.uniform(key, minval=0.1, maxval=0.4)  # probability of packet leaving

    # transition matrix
    Ph = jnp.zeros((S, A, S))

    # when s==0
    Ph = Ph.at[0, 0, 0].set(rho * mu1 + (1-rho) * (1-mu1) + rho * (1 - mu1))
    Ph = Ph.at[0, 0, 1].set((1 - rho) * mu1)
    Ph = Ph.at[0, 1, 0].set(rho * mu2 + (1-rho) * (1-mu2) + rho * (1 - mu2))
    Ph = Ph.at[0, 1, 1].set((1 - rho) * mu2)

    # when 0 < s < N
    for s in range(1, S):
        Ph = Ph.at[s, 0, s-1].set(rho * (1 - mu1))
        Ph = Ph.at[s, 0, s].set(rho * mu1 + (1-rho) * (1-mu1))
        Ph = Ph.at[s, 0, s+1].set((1 - rho) * mu1)
        Ph = Ph.at[s, 1, s-1].set(rho * (1 - mu2))
        Ph = Ph.at[s, 1, s].set(rho * mu2 + (1-rho) * (1-mu2))
        Ph = Ph.at[s, 1, s+1].set((1 - rho) * mu2)

    # when s==B
    Ph = Ph.at[buff, 0, buff].set(rho * mu1 + (1-rho) * (1-mu1) + (1 - rho) * mu1)
    Ph = Ph.at[buff, 0, buff-1].set(rho * (1 - mu1))
    Ph = Ph.at[buff, 1, buff].set(rho * mu2 + (1-rho) * (1-mu2) + (1 - rho) * mu2)
    Ph = Ph.at[buff, 1, buff-1].set(rho * (1 - mu2))

    P = jnp.tile(Ph, (H, 1, 1, 1))

    # create initial distribution
    key, _ = jax.random.split(key)
    init_dist = jnp.zeros(S)
    init_dist = init_dist.at[0].set(1.0)
    np.testing.assert_allclose(init_dist.sum(axis=-1), 1, atol=1e-6)

    cmdp = CMDP(S_set, A_set, H, d, phi, rew, utility, const, P, init_dist, xi=xi)
    return cmdp

