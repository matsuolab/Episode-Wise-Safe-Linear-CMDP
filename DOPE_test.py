import jax
import pytest
import numpy as np
from itertools import product
from scipy.optimize import linprog
from envs.cmdp import CMDP
from utils import set_cmdp_info, EvalRegQ
from DOPE import compute_optimal_policy_LP
from mpax import r2HPDHG
import importlib
import jax.numpy as jnp



@pytest.mark.parametrize("module_name", ["tabular", "streaming", "linear"])
@pytest.mark.parametrize("seed", [123, 999, 1911])
def test_compute_optimal_policy(module_name, seed):
    # Test if compute_optimal_policy_LP returns an optimal policy

    key = jax.random.PRNGKey(seed)
    module = importlib.import_module(f"envs.{module_name}")
    cmdp = module.create_cmdp(key)
    cmdp = set_cmdp_info(cmdp)

    solver = r2HPDHG(eps_abs=1e-3, eps_rel=1e-3, verbose=False)
    policy = compute_optimal_policy_LP(cmdp, solver)
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
