import pytest
from proposal import compute_softmax_pol_h, compute_softmax_pol, bisection_search_lam
from envs import tabular, streaming
from utils import set_cmdp_info, sample_and_compute_regret
import jax
import jax.numpy as jnp
import importlib
from envs.cmdp import CMDP
import numpy as np


def test_compute_softmax_pol_h_shape():
    ent_coef = 1.0
    lam = 0.5
    S, A = 5, 2
    rQ_h = jnp.ones((S, A))
    dQ_h = jnp.ones((S, A))
    uQ_h = jnp.ones((S, A))
    pol_h = compute_softmax_pol_h(rQ_h, dQ_h, uQ_h, ent_coef, lam)
    assert pol_h.shape == (S, A)
    assert jnp.allclose(jnp.sum(pol_h, axis=-1), 1.0)


@pytest.mark.parametrize("module_name", ["tabular", "streaming"])
def test_compute_softmax_pol_output(module_name):
    key = jax.random.PRNGKey(0)
    module = importlib.import_module(f"envs.{module_name}")
    cmdp = module.create_cmdp(key)
    bonus = jnp.ones((cmdp.H, cmdp.S, cmdp.A))
    ent_coef = 1.0
    lam = 0.5
    Cr, Cd, Cu, Bd = 1.0, 1.0, 1.0, 1.0
    total_util, pol = compute_softmax_pol(bonus, cmdp, ent_coef, lam, Cr, Cd, Cu, Bd)
    assert isinstance(total_util, jnp.ndarray)
    assert pol.shape[0] == bonus.shape[0]
    assert jnp.all(pol >= 0)
    assert jnp.allclose(jnp.sum(pol, axis=-1), 1.0)


@pytest.mark.parametrize("module_name", ["tabular", "streaming"])
def test_bisection_search_lam_shape(module_name):
    key = jax.random.PRNGKey(0)
    module = importlib.import_module(f"envs.{module_name}")
    cmdp = module.create_cmdp(key)

    bonus = jnp.ones((cmdp.H, cmdp.S, cmdp.A))
    ent_coef = 1.0
    Cr, Cd, Cu, Bd, Clam = 1.0, 1.0, 1.0, 1.0, 1.0
    lam_range = jnp.array([0.0, Clam], dtype=jnp.float32)
    iter_length = 5
    pol = bisection_search_lam(cmdp, lam_range, bonus, ent_coef, Cr, Cd, Cu, Bd, Clam, iter_length)
    assert pol.shape == (cmdp.H, cmdp.S, cmdp.A)
    assert jnp.all(pol >= 0)
    assert jnp.allclose(jnp.sum(pol, axis=-1), 1.0)


@pytest.mark.parametrize("module_name", ["tabular", "streaming"])
@pytest.mark.parametrize("seed", [0, 1, 42, 123, 331, 544, 803, 999, 1911])
def test_bisection_search_close_to_optimal(module_name, seed):
    key = jax.random.PRNGKey(seed)
    module = importlib.import_module(f"envs.{module_name}")
    cmdp = module.create_cmdp(key)
    cmdp = set_cmdp_info(cmdp)

    bonus = jnp.ones((cmdp.H, cmdp.S, cmdp.A))
    ent_coef = 0.01
    Cr, Cd, Cu, Bd, Clam = 0, 0, 0, 0, 1000 
    lam_range = jnp.array([0.0, Clam], dtype=jnp.float32)
    iter_length = 30
    pol = bisection_search_lam(cmdp, lam_range, bonus, ent_coef, Cr, Cd, Cu, Bd, Clam, iter_length)

    _, _, err_rew, err_vio = sample_and_compute_regret(key, cmdp, pol)

    np.testing.assert_allclose(err_rew, 0, atol=1e-2)
    np.testing.assert_allclose(err_vio, 0, atol=1e-2)


@pytest.mark.parametrize("module_name", ["tabular", "streaming"])
@pytest.mark.parametrize("seed", [0, 1, 42, 123, 331])
@pytest.mark.parametrize("Cr, Cd, Cu, Bd", [
    (0.0, 0.0, 100.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
])
def test_bisection_search_pessimistic_effect(module_name, seed, Cr, Cd, Cu, Bd):
    # this test checks if the bisection search yields pessimistic policies

    key = jax.random.PRNGKey(seed)
    module = importlib.import_module(f"envs.{module_name}")
    cmdp = module.create_cmdp(key)
    cmdp = cmdp._replace(const_scale=0.8)
    cmdp = set_cmdp_info(cmdp)

    # Add small noise to transition probabilities P
    noise = 1e-2 * np.random.randn(*cmdp.P.shape)
    P_noisy = np.clip(cmdp.P + noise, 0, 1)
    P_noisy = P_noisy / P_noisy.sum(axis=-1, keepdims=True)
    est_cmdp = cmdp._replace(P=P_noisy)
    bonus = jnp.ones((cmdp.H, cmdp.S, cmdp.A))

    ent_coef = 0.01
    Clam = 1000
    lam_range = jnp.array([0.0, Clam], dtype=jnp.float32)
    iter_length = 30
    pol = bisection_search_lam(est_cmdp, lam_range, bonus, ent_coef, Cr, Cd, Cu, Bd, Clam, iter_length)

    _, _, err_rew, err_vio = sample_and_compute_regret(key, cmdp, pol)

    if Cu == 0:
        # If Cu is zero, we expect no utility violation
        assert err_vio > 0
    else:
        np.testing.assert_allclose(err_vio, 0, atol=1e-3)