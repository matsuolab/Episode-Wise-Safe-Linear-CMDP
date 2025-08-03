import jax
import pytest
import numpy as np
from itertools import product
from scipy.optimize import linprog
from envs.cmdp import CMDP
from envs import tabular, streaming
from utils import compute_greedy_Q_utility, EvalRegQ
from utils import compute_occupancy_measure
from utils import compute_policy_matrix
from utils import compute_optimal_rew_util
from utils import set_cmdp_info
from utils import deploy_policy_episode
from utils import sample_and_compute_regret
from utils import Sherman_Morrison_update_H
from utils import compute_bonus
from utils import update_ephi_sum_and_estimate_P
import importlib

import jax.numpy as jnp

@pytest.mark.parametrize("module_name", ["tabular", "streaming", "linear"])
def test_EvalRegQ_shape(module_name):
    key = jax.random.PRNGKey(0)
    module = importlib.import_module(f"envs.{module_name}")
    cmdp = module.create_cmdp(key)
    uni_policy = jnp.ones((cmdp.H, cmdp.S, cmdp.A)) / cmdp.A
    Q = EvalRegQ(uni_policy, cmdp.rew, cmdp.P, 0.1)
    assert Q.shape == (cmdp.H, cmdp.S, cmdp.A)


@pytest.mark.parametrize("module_name", ["tabular", "streaming", "linear"])
def test_compare_uni_and_greedy_Q_values(module_name):
    key = jax.random.PRNGKey(0)
    module = importlib.import_module(f"envs.{module_name}")
    cmdp = module.create_cmdp(key)
    uni_policy = jnp.ones((cmdp.H, cmdp.S, cmdp.A)) / cmdp.A
    Q_uni = EvalRegQ(uni_policy, cmdp.utility, cmdp.P, 0.0)
    Q_greedy = compute_greedy_Q_utility(cmdp)
    assert jnp.all(Q_uni <= Q_greedy)



@pytest.mark.parametrize("module_name", ["tabular", "streaming", "linear"])
def test_EvalRegQ_regularization_effect(module_name):
    key = jax.random.PRNGKey(0)
    module = importlib.import_module(f"envs.{module_name}")
    cmdp = module.create_cmdp(key)
    uni_policy = jnp.ones((cmdp.H, cmdp.S, cmdp.A)) / cmdp.A
    Q_reg = EvalRegQ(uni_policy, cmdp.rew, cmdp.P, 0.1)
    Q_noreg = EvalRegQ(uni_policy, cmdp.rew, cmdp.P, 0)
    assert jnp.all(Q_reg >= Q_noreg)


@pytest.mark.parametrize("module_name", ["tabular", "streaming", "linear"])
def test_EvalRegQ_threshold_effect(module_name):
    key = jax.random.PRNGKey(0)
    module = importlib.import_module(f"envs.{module_name}")
    cmdp = module.create_cmdp(key)
    uni_policy = jnp.ones((cmdp.H, cmdp.S, cmdp.A)) / cmdp.A
    Q_high_thresh = EvalRegQ(uni_policy, cmdp.rew, cmdp.P, 0.1, thresh_coef=10.0)
    Q_low_thresh = EvalRegQ(uni_policy, cmdp.rew, cmdp.P, 0.1, thresh_coef=0.1)
    assert jnp.all(Q_high_thresh >= Q_low_thresh)


@pytest.mark.parametrize("module_name", ["tabular", "streaming", "linear"])
def test_compute_policy_matrix_shape(module_name):
    key = jax.random.PRNGKey(0)
    module = importlib.import_module(f"envs.{module_name}")
    cmdp = module.create_cmdp(key)
    policy = jnp.ones((cmdp.H, cmdp.S, cmdp.A)) / cmdp.A
    PI = compute_policy_matrix(policy)
    assert PI.shape == (cmdp.H, cmdp.S, cmdp.S * cmdp.A)


@pytest.mark.parametrize("module_name", ["tabular", "streaming", "linear"])
def test_compute_occupancy_measure_shape(module_name):
    key = jax.random.PRNGKey(0)
    module = importlib.import_module(f"envs.{module_name}")
    cmdp = module.create_cmdp(key)
    policy = jnp.ones((cmdp.H, cmdp.S, cmdp.A)) / cmdp.A
    occ = compute_occupancy_measure(cmdp, policy)
    assert occ.shape == (cmdp.H, cmdp.S, cmdp.A)


@pytest.mark.parametrize("module_name", ["tabular", "streaming", "linear"])
def test_compute_occupancy_measure_nonnegative(module_name):
    key = jax.random.PRNGKey(0)
    module = importlib.import_module(f"envs.{module_name}")
    cmdp = module.create_cmdp(key)
    policy = jnp.ones((cmdp.H, cmdp.S, cmdp.A)) / cmdp.A
    occ = compute_occupancy_measure(cmdp, policy)
    assert jnp.all(occ >= 0)


@pytest.mark.parametrize("module_name", ["tabular", "streaming", "linear"])
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


@pytest.mark.parametrize("module_name", ["tabular", "streaming", "linear"])
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


@pytest.mark.parametrize("module_name", ["tabular", "streaming", "linear"])
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


@pytest.mark.parametrize("module_name", ["tabular", "streaming", "linear"])
@pytest.mark.parametrize("seed", [0, 1, 42, 123, 999])
def test_set_cmdp_info(module_name, seed):
    key = jax.random.PRNGKey(seed)
    module = importlib.import_module(f"envs.{module_name}")
    cmdp = module.create_cmdp(key)
    cmdp = set_cmdp_info(cmdp)

    assert cmdp.xi > 0, "xi should be greater than 0"
    assert cmdp.const > 0, "const should be greater than 0"


@pytest.mark.parametrize("module_name", ["tabular", "streaming", "linear"])
@pytest.mark.parametrize("seed", [123, 331, 544, 803, 999, 1911])
def test_compute_optimal_rew_util(module_name, seed):
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

    key = jax.random.PRNGKey(seed)
    module = importlib.import_module(f"envs.{module_name}")
    cmdp = module.create_cmdp(key)
    cmdp = set_cmdp_info(cmdp)

    optimal_rew, optimal_util = compute_optimal_rew_util(cmdp, 0.02, 50000)
    optimal_rew_LP, optimal_util_LP = compute_optimal_rew_util_LP(cmdp)

    assert optimal_util >= cmdp.const - 0.05, f"total_utility: {optimal_util}, const: {cmdp.const}"
    np.testing.assert_allclose(optimal_rew, optimal_rew_LP, atol=0.05)


@pytest.mark.parametrize("module_name", ["tabular", "streaming", "linear"])
@pytest.mark.parametrize("seed", [0, 1, 42, 123, 999])
def test_deploy_policy_episode(module_name, seed):
    key = jax.random.PRNGKey(seed)
    module = importlib.import_module(f"envs.{module_name}")
    cmdp = module.create_cmdp(key)
    policy = jnp.ones((cmdp.H, cmdp.S, cmdp.A)) / cmdp.A

    key, subkey = jax.random.split(key)
    _, trajectory = deploy_policy_episode(cmdp, subkey, policy)

    assert trajectory.shape == (cmdp.H, 2)


@pytest.mark.parametrize("module_name", ["tabular", "streaming", "linear"])
@pytest.mark.parametrize("seed1, seed2", [(0, 1), (42, 123), (999, 1000)])
def test_deploy_policy_episode_randomness(module_name, seed1, seed2):
    key1 = jax.random.PRNGKey(seed1)
    key2 = jax.random.PRNGKey(seed2)
    module = importlib.import_module(f"envs.{module_name}")
    cmdp = module.create_cmdp(key1)
    policy = jnp.ones((cmdp.H, cmdp.S, cmdp.A)) / cmdp.A

    _, traj1 = deploy_policy_episode(cmdp, key1, policy)
    _, traj2 = deploy_policy_episode(cmdp, key2, policy)

    # With different seeds, trajectories should differ
    assert not jnp.allclose(traj1, traj2), "Trajectories should differ for different seeds"


@pytest.mark.parametrize("module_name", ["tabular", "streaming", "linear"])
def test_deploy_policy_episode_repeatability(module_name):
    seed = 12345
    key = jax.random.PRNGKey(seed)
    module = importlib.import_module(f"envs.{module_name}")
    cmdp = module.create_cmdp(key)
    policy = jnp.ones((cmdp.H, cmdp.S, cmdp.A)) / cmdp.A

    _, traj1 = deploy_policy_episode(cmdp, key, policy)
    _, traj2 = deploy_policy_episode(cmdp, key, policy)

    # With same seed and same policy, trajectories should be identical
    assert jnp.allclose(traj1, traj2), "Trajectories should be identical for same seed and policy"


@pytest.mark.parametrize("module_name", ["tabular", "streaming", "linear"])
@pytest.mark.parametrize("seed", [0, 1, 42, 123, 999])
def test_sample_and_compute_regret(module_name, seed):
    key = jax.random.PRNGKey(seed)
    module = importlib.import_module(f"envs.{module_name}")
    cmdp = module.create_cmdp(key)
    cmdp = set_cmdp_info(cmdp)

    policy = jnp.ones((cmdp.H, cmdp.S, cmdp.A)) / cmdp.A
    _, traj, err_rew, err_vio = sample_and_compute_regret(key, cmdp, policy)

    Q_rew = EvalRegQ(policy, cmdp.rew, cmdp.P, 0)

    init_dist = cmdp.init_dist
    total_rew = ((Q_rew * policy)[0].sum(axis=-1) * init_dist).sum()
    np.testing.assert_allclose(err_rew, cmdp.optimal_ret - total_rew, atol=1e-6)

    Q_utility = EvalRegQ(policy, cmdp.utility, cmdp.P, 0)
    total_util = ((Q_utility * policy)[0].sum(axis=-1) * init_dist).sum()
    np.testing.assert_allclose(jnp.maximum(cmdp.const - total_util, 0), err_vio, atol=1e-6)


@pytest.mark.parametrize("seed", [0, 1, 42, 123, 999])
def test_Sherman_Morrison_update_H(seed):
    # Test Sherman_Morrison_update_H for correctness
    H = 3
    d = 2
    key = jax.random.PRNGKey(seed)
    Lambda = jnp.tile(jnp.eye(d), (H, 1, 1))  # H x d x d
    Lambda_inv_exp = jax.vmap(jnp.linalg.inv)(Lambda)  # H x d x d
    Lambda_inv_Sh = Lambda_inv_exp.copy()

    for _ in range(10):
        key, subkey = jax.random.split(key)
        Phi = jax.random.normal(key, (H, d))
        
        Lambda_inv_Sh = Sherman_Morrison_update_H(Lambda_inv_Sh, Phi)

        phi_phi = jax.vmap(jnp.dot)(Phi.reshape(H, d, 1), Phi.reshape(H, 1, d))
        Lambda = Lambda + phi_phi
        Lambda_inv_exp = jax.vmap(jnp.linalg.inv)(Lambda)
        np.testing.assert_allclose(Lambda_inv_Sh, Lambda_inv_exp, atol=1e-5)


@pytest.mark.parametrize("module_name", ["tabular", "streaming", "linear"])
@pytest.mark.parametrize("seed", [0, 1, 42, 123, 999])
def test_compute_bonus_shape_and_nonnegative(module_name, seed):
    key = jax.random.PRNGKey(seed)
    module = importlib.import_module(f"envs.{module_name}")
    cmdp = module.create_cmdp(key)
    H, S, A, d = cmdp.H, cmdp.S, cmdp.A, cmdp.d
    Lambda_inv = jnp.tile(jnp.eye(d), (H, 1, 1))

    for _ in range(100):
        key, subkey = jax.random.split(key)
        Phi = jax.random.normal(key, (H, d))
        Lambda_inv= Sherman_Morrison_update_H(Lambda_inv, Phi)
        bonus = compute_bonus(Lambda_inv, cmdp)
        assert bonus.shape == (H, S, A)
        assert jnp.all(bonus >= 0)


@pytest.mark.parametrize("module_name", ["tabular", "linear"])
@pytest.mark.parametrize("seed", [0, 1, 42, 123, 999])
def test_update_ephi_sum_and_estimate_P(module_name, seed):

    H, S, A, d = 3, 3, 2, 2

    key = jax.random.PRNGKey(seed)
    module = importlib.import_module(f"envs.{module_name}")
    cmdp = module.create_cmdp(key, S=S, A=A, d=d, H=H)
    cmdp = set_cmdp_info(cmdp)
    uni_policy = jnp.ones((cmdp.H, cmdp.S, cmdp.A)) / cmdp.A
    ephi_sum = jnp.zeros((cmdp.H, cmdp.S, cmdp.d))
    Lambda = jnp.tile(jnp.eye(cmdp.d), (cmdp.H, 1, 1))  # H x d x d
    Lambda_inv = jax.vmap(jnp.linalg.inv)(Lambda)

    est_P_errors = []
    for _ in range(100):
        key, subkey = jax.random.split(key)
        traj = sample_and_compute_regret(key, cmdp, uni_policy)[1]
        Phi = cmdp.phi[traj[:, 0]]  # (H x d)
        Lambda_inv = Sherman_Morrison_update_H(Lambda_inv, Phi)

        ephi_sum, est_P = update_ephi_sum_and_estimate_P(ephi_sum, Lambda_inv, traj, cmdp)

        est_P_errors.append(np.linalg.norm(est_P - cmdp.P))

    assert est_P_errors[0] > est_P_errors[-1], "Estimated P should improve over iterations"


@pytest.mark.parametrize("module_name", ["tabular", "streaming", "linear"])
@pytest.mark.parametrize("seed", [0, 1, 42, 123, 999])
def test_compute_bonus_decrease(module_name, seed):
    key = jax.random.PRNGKey(seed)
    module = importlib.import_module(f"envs.{module_name}")
    cmdp = module.create_cmdp(key)
    cmdp = set_cmdp_info(cmdp)
    H, S, A, d = cmdp.H, cmdp.S, cmdp.A, cmdp.d
    Lambda_inv = jnp.tile(jnp.eye(d), (H, 1, 1))

    bonus_initial = compute_bonus(Lambda_inv, cmdp)
    uni_policy = jnp.ones((cmdp.H, cmdp.S, cmdp.A)) / cmdp.A

    for _ in range(10):
        key, subkey = jax.random.split(key)
        traj = sample_and_compute_regret(key, cmdp, uni_policy)[1]
        Phi = cmdp.phi[traj[:, 0]]  # (H x d)
        Lambda_inv = Sherman_Morrison_update_H(Lambda_inv, Phi)
        bonus = compute_bonus(Lambda_inv, cmdp)

    assert jnp.all(bonus <= bonus_initial), "Bonus should decrease"