import pytest
import jax
from envs import tabular, streaming, linear
from envs.cmdp import CMDP
import importlib

import jax.numpy as jnp


@pytest.mark.parametrize("module_name", ["tabular", "streaming", "linear"])
def test_create_cmdp_output_type(module_name):
    key = jax.random.PRNGKey(0)
    module = importlib.import_module(f"envs.{module_name}")
    cmdp = module.create_cmdp(key)
    assert isinstance(cmdp, CMDP)


@pytest.mark.parametrize("module_name", ["tabular", "streaming", "linear"])
def test_create_cmdp_maximum_rew_utility(module_name):
    key = jax.random.PRNGKey(0)
    module = importlib.import_module(f"envs.{module_name}")
    cmdp = module.create_cmdp(key)
    assert cmdp.rew.max() <= 1.0
    assert cmdp.utility.max() <= 1.0


@pytest.mark.parametrize("module_name", ["tabular", "streaming", "linear"])
def test_create_cmdp_shapes_and_values(module_name):
    key = jax.random.PRNGKey(42)
    module = importlib.import_module(f"envs.{module_name}")
    cmdp = module.create_cmdp(key)

    assert cmdp.rew.shape == (cmdp.H, cmdp.S, cmdp.A)
    assert cmdp.utility.shape == (cmdp.H, cmdp.S, cmdp.A)
    assert cmdp.P.shape == (cmdp.H, cmdp.S, cmdp.A, cmdp.S)
    assert cmdp.init_dist.shape == (cmdp.S,)
    assert cmdp.phi.shape == (cmdp.S * cmdp.A, cmdp.d)

    assert jnp.all(cmdp.rew >= 0)
    assert jnp.all(cmdp.utility >= 0)
    assert jnp.all(cmdp.P >= 0)
    assert jnp.all(cmdp.init_dist >= 0)

    assert jnp.allclose(cmdp.P.sum(axis=-1), 1, atol=1e-6)
    assert jnp.isclose(cmdp.init_dist.sum(), 1, atol=1e-6)

    assert cmdp.rew.max() > 0
    assert cmdp.utility.max() > 0


@pytest.mark.parametrize("module_name", ["tabular", "streaming", "linear"])
def test_create_cmdp_randomness(module_name):
    key1 = jax.random.PRNGKey(123)
    key2 = jax.random.PRNGKey(456)
    module = importlib.import_module(f"envs.{module_name}")
    cmdp1 = module.create_cmdp(key1)
    cmdp2 = module.create_cmdp(key2)
    if module_name == "tabular":
        assert not jnp.allclose(cmdp1.rew, cmdp2.rew)
        assert not jnp.allclose(cmdp1.utility, cmdp2.utility)
    assert not jnp.allclose(cmdp1.P, cmdp2.P)
