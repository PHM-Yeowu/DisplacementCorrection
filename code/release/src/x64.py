"""Run jaxpi2 in float64 without editing it.

Enabling x64 is not enough on its own: flax.linen.Dense stores float32
parameters by default, optax.contrib.schedule_free may keep its iterate in
float32, and soap_jax keeps its preconditioner (GG, Q) in float32. Import this
module before jaxpi; `check_dtypes` lists any float32 leaf left in a train state.
"""

import os

os.environ["JAX_ENABLE_X64"] = "1"

from typing import Any

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

import jax.numpy as jnp
import flax.linen as nn
import optax


class Dense(nn.Dense):
    # same class name, so parameter paths (Dense_0, ...) match the float32 runs
    param_dtype: Any = jnp.float64


nn.Dense = Dense

_schedule_free = optax.contrib.schedule_free


def schedule_free(*args, **kwargs):
    kwargs.setdefault("state_dtype", jnp.float64)
    return _schedule_free(*args, **kwargs)


optax.contrib.schedule_free = schedule_free

import soap_jax

_soap = soap_jax.soap


def soap(*args, **kwargs):
    kwargs.setdefault("qr_dtype", jnp.float64)
    return _soap(*args, **kwargs)


soap_jax.soap = soap  # jaxpi.models binds it with `from soap_jax import soap`


def check_dtypes(tree):
    """Paths of the floating leaves of `tree` that are not float64."""
    leaves = jax.tree_util.tree_leaves_with_path(tree)
    return [
        jax.tree_util.keystr(path)
        for path, leaf in leaves
        if hasattr(leaf, "dtype")
        and jnp.issubdtype(leaf.dtype, jnp.floating)
        and leaf.dtype != jnp.float64
    ]
