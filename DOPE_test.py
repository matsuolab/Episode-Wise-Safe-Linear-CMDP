import jax
import pytest
import numpy as np
from itertools import product
from envs.cmdp import CMDP
from utils import set_cmdp_info, EvalRegQ
from DOPE import compute_extended_optimal_policy, build_lp_jax_sparse
from mpax import r2HPDHG
import importlib
import jax.numpy as jnp


@pytest.mark.parametrize("module_name", ["tabular", "streaming", "linear"])
@pytest.mark.parametrize("seed", [123, 999, 1911])
def test_build_lp_jax_sparse(module_name, seed):
    key = jax.random.PRNGKey(seed)
    module = importlib.import_module(f"envs.{module_name}")
    cmdp = module.create_cmdp(key)
    cmdp = set_cmdp_info(cmdp)

    H, S, A = cmdp.rew.shape
    bonus = jnp.zeros((H, S, A, S), dtype=jnp.float32)  # if there is no bonus, it should returns the same policy as the optimal one

    c, A, b, G, h = build_lp_jax_sparse(-cmdp.rew, cmdp.init_dist, cmdp.P, bonus, cmdp.utility, cmdp.const)
    

@pytest.mark.parametrize("module_name", ["tabular", "streaming", "linear"])
@pytest.mark.parametrize("seed", [123, 999, 1911])
def test_extended_optimal_policy(module_name, seed):
    # Test if compute_extended_optimal_policy returns an optimal policy

    key = jax.random.PRNGKey(seed)
    module = importlib.import_module(f"envs.{module_name}")
    cmdp = module.create_cmdp(key)
    cmdp = set_cmdp_info(cmdp)

    H, S, A = cmdp.rew.shape
    bonus = jnp.zeros((H, S, A, S), dtype=jnp.float32)  # if there is no bonus, it should returns the same policy as the optimal one
    policy = compute_extended_optimal_policy(cmdp, bonus)
    np.testing.assert_allclose(policy.sum(axis=-1), 1.0, atol=1e-5)

    # compute temporal regret
    Q_rew= EvalRegQ(policy, cmdp.rew, cmdp.P, 0)
    Q_utility = EvalRegQ(policy, cmdp.utility, cmdp.P, 0)
    init_dist = cmdp.init_dist
    total_rew = ((Q_rew * policy)[0].sum(axis=-1) * init_dist).sum()
    total_utility = ((Q_utility * policy)[0].sum(axis=-1) * init_dist).sum()
    
    err_rew = jnp.maximum(cmdp.optimal_ret - total_rew, 0)
    err_vio = jnp.maximum(cmdp.const - total_utility, 0)

    assert err_rew < 1e-2, f"Expected low regret for rewards, got {err_rew}"
    assert err_vio < 1e-2, f"Expected low violation, got {err_vio}"
