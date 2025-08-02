import jax
import pytest
import numpy as np
from itertools import product
from scipy.optimize import linprog
from envs.cmdp import CMDP
from envs import tabular, streaming
from typing import NamedTuple
from utils import compute_greedy_Q_utility, EvalRegQ
from utils import compute_occupancy_measure
from utils import compute_policy_matrix
from utils import compute_optimal_rew_util
import importlib

import jax.numpy as jnp

@pytest.mark.parametrize("module_name", ["tabular", "streaming"])
def test_EvalRegQ_shape(module_name):
    key = jax.random.PRNGKey(0)
    module = importlib.import_module(f"envs.{module_name}")
    cmdp = module.create_cmdp(key)
    uni_policy = jnp.ones((cmdp.H, cmdp.S, cmdp.A)) / cmdp.A
    Q = EvalRegQ(uni_policy, cmdp.rew, cmdp.P, 0.1)
    assert Q.shape == (cmdp.H, cmdp.S, cmdp.A)


@pytest.mark.parametrize("module_name", ["tabular", "streaming"])
def test_compare_uni_and_greedy_Q_values(module_name):
    key = jax.random.PRNGKey(0)
    module = importlib.import_module(f"envs.{module_name}")
    cmdp = module.create_cmdp(key)
    uni_policy = jnp.ones((cmdp.H, cmdp.S, cmdp.A)) / cmdp.A
    Q_uni = EvalRegQ(uni_policy, cmdp.utility, cmdp.P, 0.0)
    Q_greedy = compute_greedy_Q_utility(cmdp)
    assert jnp.all(Q_uni <= Q_greedy)


@pytest.mark.parametrize("module_name", ["tabular", "streaming"])
def test_EvalRegQ_regularization_effect(module_name):
    key = jax.random.PRNGKey(0)
    module = importlib.import_module(f"envs.{module_name}")
    cmdp = module.create_cmdp(key)
    uni_policy = jnp.ones((cmdp.H, cmdp.S, cmdp.A)) / cmdp.A
    Q_reg = EvalRegQ(uni_policy, cmdp.rew, cmdp.P, 0.1)
    Q_noreg = EvalRegQ(uni_policy, cmdp.rew, cmdp.P, 0)
    assert jnp.all(Q_reg >= Q_noreg)


@pytest.mark.parametrize("module_name", ["tabular", "streaming"])
def test_EvalRegQ_threshold_effect(module_name):
    key = jax.random.PRNGKey(0)
    module = importlib.import_module(f"envs.{module_name}")
    cmdp = module.create_cmdp(key)
    uni_policy = jnp.ones((cmdp.H, cmdp.S, cmdp.A)) / cmdp.A
    Q_high_thresh = EvalRegQ(uni_policy, cmdp.rew, cmdp.P, 0.1, thresh_coef=10.0)
    Q_low_thresh = EvalRegQ(uni_policy, cmdp.rew, cmdp.P, 0.1, thresh_coef=0.1)
    assert jnp.all(Q_high_thresh >= Q_low_thresh)


@pytest.mark.parametrize("module_name", ["tabular", "streaming"])
def test_compute_policy_matrix_shape(module_name):
    key = jax.random.PRNGKey(0)
    module = importlib.import_module(f"envs.{module_name}")
    cmdp = module.create_cmdp(key)
    policy = jnp.ones((cmdp.H, cmdp.S, cmdp.A)) / cmdp.A
    PI = compute_policy_matrix(policy)
    assert PI.shape == (cmdp.H, cmdp.S, cmdp.S * cmdp.A)


@pytest.mark.parametrize("module_name", ["tabular", "streaming"])
def test_compute_occupancy_measure_shape(module_name):
    key = jax.random.PRNGKey(0)
    module = importlib.import_module(f"envs.{module_name}")
    cmdp = module.create_cmdp(key)
    policy = jnp.ones((cmdp.H, cmdp.S, cmdp.A)) / cmdp.A
    occ = compute_occupancy_measure(cmdp, policy)
    assert occ.shape == (cmdp.H, cmdp.S, cmdp.A)


@pytest.mark.parametrize("module_name", ["tabular", "streaming"])
def test_compute_occupancy_measure_nonnegative(module_name):
    key = jax.random.PRNGKey(0)
    module = importlib.import_module(f"envs.{module_name}")
    cmdp = module.create_cmdp(key)
    policy = jnp.ones((cmdp.H, cmdp.S, cmdp.A)) / cmdp.A
    occ = compute_occupancy_measure(cmdp, policy)
    assert jnp.all(occ >= 0)


@pytest.mark.parametrize("module_name", ["tabular", "streaming"])
def test_compute_occupancy_measure_sum(module_name):
    key = jax.random.PRNGKey(0)
    module = importlib.import_module(f"envs.{module_name}")
    cmdp = module.create_cmdp(key)
    policy = jnp.ones((cmdp.H, cmdp.S, cmdp.A)) / cmdp.A
    occ = compute_occupancy_measure(cmdp, policy)
    # The sum of occupancy at each time step should equal to 1
    for h in range(cmdp.H):
        occ_sum = occ[h].sum()
        assert jnp.allclose(occ_sum, 1.0, atol=1e-6), f"Occupancy sum at time {h} does not equal 1: {occ_sum}"


@pytest.mark.parametrize("module_name", ["tabular", "streaming"])
def test_compute_occupancy_measure_sum_conservation(module_name):
    key = jax.random.PRNGKey(0)
    module = importlib.import_module(f"envs.{module_name}")
    cmdp = module.create_cmdp(key)
    policy = jnp.ones((cmdp.H, cmdp.S, cmdp.A)) / cmdp.A
    occ = compute_occupancy_measure(cmdp, policy)
    # The sum of occupancy at h=0 should match the initial distribution
    occ0_sum = occ[0].sum()
    init_sum = cmdp.init_dist.sum() if hasattr(cmdp, "init_dist") else 1.0
    np.testing.assert_allclose(float(occ0_sum), float(init_sum), rtol=1e-5)


@pytest.mark.parametrize("module_name", ["tabular", "streaming"])
def test_compute_occupancy_measure_policy_effect(module_name):
    key = jax.random.PRNGKey(0)
    module = importlib.import_module(f"envs.{module_name}")
    cmdp = module.create_cmdp(key)
    # Uniform policy
    policy_uni = jnp.ones((cmdp.H, cmdp.S, cmdp.A)) / cmdp.A
    occ_uni = compute_occupancy_measure(cmdp, policy_uni)
    # Greedy policy (all actions on first action)
    policy_greedy = jnp.zeros((cmdp.H, cmdp.S, cmdp.A))
    policy_greedy = policy_greedy.at[..., 0].set(1)
    occ_greedy = compute_occupancy_measure(cmdp, policy_greedy)
    # Should be different
    assert not jnp.allclose(occ_uni, occ_greedy)


@pytest.mark.parametrize("module_name", ["tabular", "streaming"])
def test_compute_optimal_rew_util(module_name):
    def compute_optimal_rew_util_LP(cmdp):
        H, S, A = cmdp.rew.shape
        B = np.zeros((H, S, A, H, S, A))
        # Initial occupancy constraints
        for s, a in product(range(S), range(A)):
            B[0, s, a, 0, s] = 1
        # Flow conservation constraints
        for h in range(1, H):
            for s, a in product(range(S), range(A)):
                B[h, s, a, h, s] = 1
                B[h, s, a, h-1] = -cmdp.P[h-1, :, :, s]
        B = B.reshape((H*S*A, H*S*A))
        mu = np.repeat(cmdp.init_dist[:, None], A, axis=1).reshape(-1)
        b = np.concatenate([mu, np.zeros((H-1)*S*A)])
        U = cmdp.utility.reshape(1, -1)
        u = np.array([cmdp.const])
        r = -cmdp.rew.reshape(-1)
        lin_res = linprog(r, A_eq=B, b_eq=b, bounds=(0, None), A_ub=-U, b_ub=-u)
        d_arr = lin_res.x.reshape(H, S, A)
        np.testing.assert_allclose(d_arr.sum(axis=(1, 2)), 1.0, atol=1e-4)
        return (d_arr * cmdp.rew).sum(), (d_arr * cmdp.utility).sum()

    key = jax.random.PRNGKey(0)
    module = importlib.import_module(f"envs.{module_name}")
    cmdp = module.create_cmdp(key)
    optimal_rew, optimal_util = compute_optimal_rew_util(cmdp)
    optimal_rew_LP, optimal_util_LP = compute_optimal_rew_util_LP(cmdp)

    np.testing.assert_allclose(optimal_rew, optimal_rew_LP, atol=1e-1)
    np.testing.assert_allclose(optimal_util, optimal_util_LP, atol=1e-1)
    assert optimal_util >= cmdp.const, f"total_utility: {optimal_util}, const: {cmdp.const}"
