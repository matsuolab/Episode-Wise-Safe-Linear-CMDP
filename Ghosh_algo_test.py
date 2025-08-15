import pytest
from Ghosh_algo import compute_softmax_pol_h_Ghosh, compute_softmax_pol_Ghosh, bisection_search_lam_Ghosh
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
    uQ_h = jnp.ones((S, A))
    pol_h = compute_softmax_pol_h_Ghosh(rQ_h, uQ_h, ent_coef, lam)
    assert pol_h.shape == (S, A)
    assert jnp.allclose(jnp.sum(pol_h, axis=-1), 1.0)


@pytest.mark.parametrize("module_name", ["tabular", "streaming", "linear"])
def test_compute_softmax_pol_output(module_name):
    key = jax.random.PRNGKey(0)
    module = importlib.import_module(f"envs.{module_name}")
    cmdp = module.create_cmdp(key)
    bonus = jnp.ones((cmdp.H, cmdp.S, cmdp.A))
    ent_coef = 1.0
    lam = 0.5
    Cr, Cu = 1.0, 1.0
    total_util, pol = compute_softmax_pol_Ghosh(bonus, cmdp, ent_coef, lam, Cr, Cu)
    assert isinstance(total_util, jnp.ndarray)
    assert pol.shape[0] == bonus.shape[0]
    assert jnp.all(pol >= 0)
    assert jnp.allclose(jnp.sum(pol, axis=-1), 1.0)


@pytest.mark.parametrize("module_name", ["tabular", "streaming", "linear"])
def test_bisection_search_lam_shape(module_name):
    key = jax.random.PRNGKey(0)
    module = importlib.import_module(f"envs.{module_name}")
    cmdp = module.create_cmdp(key)

    bonus = jnp.ones((cmdp.H, cmdp.S, cmdp.A))
    ent_coef = 1.0
    Cr, Cu, Clam = 1.0, 1.0, 1.0
    iter_length = 5
    lam_range = bisection_search_lam_Ghosh(cmdp, bonus, ent_coef, Cr, Cu, Clam, iter_length)
    _, pol = compute_softmax_pol_Ghosh(bonus, cmdp, ent_coef, lam_range[1], Cr, Cu)
    assert pol.shape == (cmdp.H, cmdp.S, cmdp.A)
    assert jnp.all(pol >= 0)
    assert jnp.allclose(jnp.sum(pol, axis=-1), 1.0)
