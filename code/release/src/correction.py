"""Corrections of a trained PINN (the backbone), in float64.

The backbone U is frozen. Its error e = U - u satisfies N(U) - N(U - e) = r with
r = N(U). A second network learns E = e / s, either additively (E = n) or as a
displacement of U plus a remainder (E = G w + b, G = U_x / sigma). It is
trained on the residual of the corrected field, N(U - s E) / s (the composite
loss of multi-stage networks). Stages repeat this about the corrected field
U - s E.

What the second network needs from U (its jet on a pool of collocation points,
r, the initial error, U on the reference grid) is computed once per stage.
"""

from functools import partial
from math import comb

import jax
import jax.numpy as jnp
from jax import jit, random, vmap
from jax.experimental.jet import jet

from jaxpi.evaluator import BaseEvaluator
from jaxpi.models import ForwardIVP


def jets(f, m):
    """(t, x) -> rows (f, f_t, f_x, ..., d^m f / dx^m) of f(t, x)."""

    def g(t, x):
        f0, f_t = jax.jvp(lambda s: f(s, x), (t,), (jnp.ones_like(t),))
        series = [[jnp.ones_like(x)] + [jnp.zeros_like(x)] * (m - 1)]
        _, f_x = jet(lambda y: f(t, y), (x,), series)
        return jnp.stack([f0, f_t, *f_x])

    return g


def mixed(f):
    """(t, x) -> d^2 f / dx dt."""

    def f_x(t, x):
        return jax.jvp(lambda y: f(t, y), (x,), (jnp.ones_like(x),))[1]

    return lambda t, x: jax.jvp(lambda s: f_x(s, x), (t,), (jnp.ones_like(t),))[1]


def dx(J, k):
    """k-th x-derivative from a jet (value, t, x, xx, ...)."""
    return J[0] if k == 0 else J[1 + k]


def displacement_jet(G, q, m):
    """Jet of E = G w + b from the jets of G and of q = (w, b) (Leibniz rule)."""
    w, b = q[:, 0], q[:, 1]
    E = [G[0] * w[0] + b[0], G[1] * w[0] + G[0] * w[1] + b[1]]
    for k in range(1, m + 1):
        E.append(sum(comb(k, j) * dx(G, k - j) * dx(w, j) for j in range(k + 1)) + dx(b, k))
    return jnp.stack(E)


def corrected(U, apply_fn, params, s, sigma, disp):
    """The field U - s E after one stage."""

    def U1(t, x):
        q = apply_fn(params, jnp.stack([t, x]))
        if disp:
            u, u_x = jax.jvp(lambda y: U(t, y), (x,), (jnp.ones_like(x),))
            return u - s * (u_x / sigma * q[0] + q[1])
        return U(t, x) - s * q[0]

    return U1


def prepare(U, R, m, u_ref, t_star, x_star, n_pool, key, chunk=8192):
    """Pool, initial error and reference-grid data for a correction of U.

    Pool rows: t, x, (u, u_t, u_x, ..., d^m u / dx^m), d^(m+1) u / dx^(m+1), u_xt.
    """
    kt, kx = random.split(key)
    t = random.uniform(kt, (n_pool,), minval=t_star[0], maxval=t_star[-1])
    x = random.uniform(kx, (n_pool,), minval=x_star[0], maxval=x_star[-1])

    jet_fn = jit(vmap(jets(U, m + 1)))
    mix_fn = jit(vmap(mixed(U)))
    J, Uxt = [], []
    for i in range(0, n_pool, chunk):
        J.append(jet_fn(t[i:i + chunk], x[i:i + chunk]))  # (chunk, m + 3)
        Uxt.append(mix_fn(t[i:i + chunk], x[i:i + chunk]))
    J, Uxt = jnp.concatenate(J), jnp.concatenate(Uxt)
    r = vmap(R)(J[:, : m + 2])

    t0 = t_star[0]
    u_ic, ux_ic = vmap(lambda y: jax.jvp(lambda z: U(t0, z), (y,), (jnp.ones_like(y),)))(x_star)
    e0 = u_ic - u_ref[0]

    s = jnp.maximum(jnp.sqrt(jnp.mean(r**2)), jnp.sqrt(jnp.mean(e0**2)))
    sigma = jnp.sqrt(jnp.mean(J[:, 2] ** 2))
    pool = jnp.concatenate([t[:, None], x[:, None], J, Uxt[:, None]], 1)

    TT, XX = jnp.meshgrid(t_star, x_star, indexing="ij")
    grid_fn = jit(vmap(lambda a, b: jax.jvp(lambda z: U(a, z), (b,), (jnp.ones_like(b),))))
    Ug, Uxg = [], []
    for i in range(0, TT.size, chunk):
        u, u_x = grid_fn(TT.ravel()[i:i + chunk], XX.ravel()[i:i + chunk])
        Ug.append(u)
        Uxg.append(u_x)
    grid = dict(t=TT.ravel(), x=XX.ravel(), U=jnp.concatenate(Ug), U_x=jnp.concatenate(Uxg))

    ic = dict(t0=t0, x=x_star, U_x=ux_ic, e0=e0)
    return dict(pool=pool, ic=ic, grid=grid, s=s, sigma=sigma, r_rms=jnp.sqrt(jnp.mean(r**2)))


class PoolSampler:
    """Random batches of pool rows."""

    def __init__(self, pool, batch_size, seed=0):
        self.pool = pool
        self.batch_size = batch_size
        self.key = random.PRNGKey(seed)

    def __iter__(self):
        return self

    def __next__(self):
        self.key, batch = _draw(self.key, self.pool, self.batch_size)
        return batch


@partial(jax.jit, static_argnums=(2,))
def _draw(key, pool, batch_size):
    """One dispatch per batch (the same values as splitting, drawing and gathering one by one)."""
    key, sub = random.split(key)
    return key, pool[random.randint(sub, (batch_size,), 0, pool.shape[0])]


class Correction(ForwardIVP):
    """Second network for E = e / s, trained on a precomputed pool of U's jets."""

    def __init__(self, config, lr, tx, arch, state, R, m, data, refs):
        super().__init__(config, lr, tx, arch, state)
        self.R, self.m = R, m
        self.disp = config.correction.form == "disp"
        self.s, self.sigma = data["s"], data["sigma"]

        ic = data["ic"]
        self.t0, self.x_ic = ic["t0"], ic["x"]
        self.G_ic, self.e0 = ic["U_x"] / self.sigma, ic["e0"] / self.s

        g = data["grid"]
        self.grid_t, self.grid_x, self.grid_G = g["t"], g["x"], g["U_x"] / self.sigma
        self.U_grid = g["U"]
        self.refs = {k: v.ravel() for k, v in refs.items()}  # reference solutions, flattened grid

    def neural_net(self, params, t, x):
        return self.state.apply_fn(params, jnp.stack([t, x]))

    def e_value(self, params, t, x, G):
        q = self.neural_net(params, t, x)
        return G * q[0] + q[1] if self.disp else q[0]

    def res(self, params, row):
        m = self.m
        U = row[2 : m + 4]
        q = jets(lambda t, x: self.neural_net(params, t, x), m)(row[0], row[1])
        if self.disp:
            # jet of G = U_x / sigma: (U_x, U_xt, U_xx, ..., d^(m+1) U / dx^(m+1)) / sigma
            G = jnp.concatenate([row[4:5], row[m + 5 : m + 6], row[5 : m + 5]]) / self.sigma
            E = displacement_jet(G, q, m)
        else:
            E = q[:, 0]
        return -self.R(U - self.s * E) / self.s

    @partial(jit, static_argnums=(0,))
    def losses(self, params, state, batch):
        res = vmap(self.res, (None, 0))(params, batch)
        e = vmap(self.e_value, (None, None, 0, 0))(params, self.t0, self.x_ic, self.G_ic)
        return {"ics": jnp.mean((e - self.e0) ** 2), "res": jnp.mean(res**2)}

    @partial(jit, static_argnums=(0,))
    def correction_on_grid(self, params):
        """s E on the reference grid (flattened)."""
        E = vmap(self.e_value, (None, 0, 0, 0))(params, self.grid_t, self.grid_x, self.grid_G)
        return self.s * E

    def errors(self, params):
        u = self.U_grid - self.correction_on_grid(params)
        return {f"error/{k}": jnp.linalg.norm(u - ref) / jnp.linalg.norm(ref) for k, ref in self.refs.items()}


class CorrectionEvaluator(BaseEvaluator):
    def __call__(self, model, state, loss_dict, batch, *args):
        self.log_dict = super().__call__(model, state, loss_dict, batch)
        self.log_dict.update(model.errors(state.params))
        return self.log_dict
