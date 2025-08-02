import jax.numpy as jnp
from typing import NamedTuple, Optional


class CMDP(NamedTuple):
    S_set: jnp.array  # state space
    A_set: jnp.array  # action space
    H: int  # horizon
    d: int  # feature map dimension
    phi: jnp.array  # feature map
    rew: jnp.array  # reward function
    utility: jnp.array  # reward for constraint
    const: float  # constraint threshold
    P: jnp.array  # transition probability kernel
    init_dist: jnp.array  # initial distribution
    xi: float  # safety parameter
    safe_policy: Optional[jnp.array] = None
    optimal_ret: Optional[jnp.array] = None  # optimal return

    @property
    def S(self) -> int:  # state space size
        return len(self.S_set)

    @property
    def A(self) -> int:  # action space size
        return len(self.A_set)
