import jax
import pytest
import numpy as np
from itertools import product
from envs.cmdp import CMDP
from utils import set_cmdp_info, EvalRegQ, sample_and_compute_regret, add_count_HSAS
from DOPE import compute_extended_optimal_policy, build_lp_jax_sparse, compute_bonus_HSAS
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
    

@pytest.mark.parametrize("module_name", ["tabular", "streaming"])
@pytest.mark.parametrize("seed", [123, 999, 1911])
def test_extended_optimal_policy(module_name, seed):
    # Test if compute_extended_optimal_policy returns an optimal policy

    key = jax.random.PRNGKey(seed)
    module = importlib.import_module(f"envs.{module_name}")
    cmdp = module.create_cmdp(key)
    cmdp = set_cmdp_info(cmdp)

    H, S, A = cmdp.rew.shape
    bonus = jnp.zeros((H, S, A, S), dtype=jnp.float32)  # if there is no bonus, it should returns the same policy as the optimal one
    policy = compute_extended_optimal_policy(cmdp, bonus, cmdp.safe_policy)
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


@pytest.mark.parametrize("module_name", ["tabular", "streaming", "linear"])
@pytest.mark.parametrize("seed", [123, 999, 1911])
def test_extended_optimal_policy_returns_safe_policy_if_infeasible(module_name, seed):
    # Test if compute_extended_optimal_policy returns the safe policy if the LP is infeasible

    key = jax.random.PRNGKey(seed)
    module = importlib.import_module(f"envs.{module_name}")
    cmdp = module.create_cmdp(key)
    cmdp = set_cmdp_info(cmdp)

    H, S, A = cmdp.rew.shape
    bonus = jnp.ones((H, S, A, S), dtype=jnp.float32)
    bonus_cmdp = cmdp._replace(utility=cmdp.utility - 100 * bonus.sum(axis=-1))
    policy = compute_extended_optimal_policy(bonus_cmdp, bonus, cmdp.safe_policy)
    np.testing.assert_allclose(policy, cmdp.safe_policy, atol=1e-5)


def test_compute_bonus_HSAS():
    # Test if compute_bonus returns a valid bonus

    H, S, A = 2, 3, 2
    count_HSAS = jnp.zeros((H, S, A, S), dtype=jnp.int32)
    count_HSAS = count_HSAS.at[0, 1, 0, 1].add(2)
    count_HSAS = count_HSAS.at[1, 2, 1, 2].add(3)

    bonus = compute_bonus_HSAS(count_HSAS)
    
    # check non-negativity and shape
    assert jnp.all(bonus >= 0), "Bonus should be non-negative"
    assert bonus.shape == (H, S, A, S), "Bonus shape mismatch"

    # check if the bonus at [0, 1, 0, 1] is smaller than [0, 0, 1, 2]
    assert bonus[0, 1, 0, 1] < bonus[0, 0, 1, 2], "Bonus should be smaller for less visited states"
   


@pytest.mark.parametrize("module_name", ["tabular", "streaming"])
def test_DOPE_update(module_name):
    # Test if compute_extended_optimal_policy returns an optimal policy

    key = jax.random.PRNGKey(0)
    module = importlib.import_module(f"envs.{module_name}")
    cmdp = module.create_cmdp(key)
    cmdp = set_cmdp_info(cmdp)

    H, S, A = cmdp.rew.shape
    pol = jnp.ones((H, S, A)) / A
    count_HSAS = jnp.zeros((H, S, A, S), dtype=jnp.int32)

    key, traj, err_rew, err_vio = sample_and_compute_regret(key, cmdp, pol)

    # update count_HSAS and compute bonus
    count_HSAS = add_count_HSAS(traj, count_HSAS)
    bonus_HSAS = compute_bonus_HSAS(count_HSAS)  # (H x S x A)
    bonus = bonus_HSAS.sum(axis=-1)

    # estimate P
    H, S, A = cmdp.rew.shape
    count_HSA = jnp.maximum(count_HSAS.sum(axis=-1, keepdims=True), 1)
    est_P = jnp.where(count_HSAS.sum(axis=-1, keepdims=True) > 0, count_HSAS / count_HSA, 1 / S)

    # compute extended LP
    bonus_rew = cmdp.rew + 0.1 * bonus
    bonus_util = cmdp.utility - 0.1 * bonus
    est_cmdp = cmdp._replace(P=est_P, rew=bonus_rew, utility=bonus_util)

    pol = compute_extended_optimal_policy(est_cmdp, bonus_HSAS, cmdp.safe_policy)

    assert bonus.shape == (H, S, A)