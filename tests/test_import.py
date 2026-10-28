"""Smoke tests for the package skeleton and the scientific stack."""

import gn_next_gen
from gn_next_gen import paths


def test_version():
    assert gn_next_gen.__version__


def test_repo_paths_exist():
    for p in (paths.DATA_RAW, paths.MANIFEST, paths.LEGACY, paths.DOCS):
        assert p.exists(), p


def test_jax_and_numpyro_import():
    import jax
    import jax.numpy as jnp
    import numpyro  # noqa: F401

    assert float(jnp.ones(3).sum()) == 3.0
    assert jax.devices()
