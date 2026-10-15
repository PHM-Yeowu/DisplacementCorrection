"""The jaxpi2 benchmarks, as the corrections need them.

For each example: the order m of its x-derivatives, its residual written on a
jet J = (u, u_t, u_x, u_xx, ..., d^m u / dx^m) at one point, and the model and
reference data exactly as the example's main.py builds them. R applied to the
jet of a network reproduces the example's r_net.

The example modules (`models`, `utils`) are imported from the example folder,
so a process works with one example only. kdv and burgers are the jaxpi2
examples; burgers_nu3 (code/release/examples) is a copy of the jaxpi2 burgers
example with the viscosity 0.003 / pi and its own reference data.
"""

import json
import os
import sys

import jax.numpy as jnp
import ml_collections

from jaxpi.models import create_model

JAXPI2 = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "external", "jaxpi2")
)
ADDED = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "examples"))
NU = {"burgers": 0.01, "burgers_nu3": 0.003}  # viscosity times pi


def example_dir(example):
    """The example folder: jaxpi2's own, or one added in code/release/examples."""
    own = os.path.join(JAXPI2, "examples", example)
    return own if os.path.isdir(own) else os.path.join(ADDED, example)


def residual(example, config):
    """(m, R) for the example."""
    if example == "kdv":
        return 3, lambda J: J[1] + J[0] * J[2] + 0.022**2 * J[4]
    if example in NU:
        nu = NU[example] / jnp.pi
        return 2, lambda J: J[1] + J[0] * J[2] - nu * J[3]
    raise NotImplementedError(example)


def enter(example):
    """Make the example's own modules importable (models, utils, configs)."""
    exdir = example_dir(example)
    os.chdir(exdir)
    sys.path.insert(0, exdir)


def _tuples(d):
    if isinstance(d, dict):
        return {k: _tuples(v) for k, v in d.items()}
    if isinstance(d, list):
        return tuple(_tuples(v) for v in d)
    return d


def load_config(run_dir):
    """The backbone's config, as its run saved it (run_jaxpi2.py)."""
    with open(os.path.join(run_dir, "config.json")) as f:
        return ml_collections.ConfigDict(_tuples(json.load(f)))


def build(example, config):
    """Model, reference solution u_ref[t, x], t_star and x_star as in main.py."""
    import models
    import utils

    u_ref, t_star, x_star = utils.get_dataset()
    cls = models.KDV if example == "kdv" else models.Burgers
    model = create_model(config, cls, u0=u_ref[0, :], t_star=t_star, x_star=x_star)
    return model, jnp.asarray(u_ref), jnp.asarray(t_star), jnp.asarray(x_star)
