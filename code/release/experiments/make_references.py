"""Converged reference solution of the jaxpi2 burgers example on its grid.

    python make_references.py burgers <out_dir>

The released reference (chebfun spin with its default ETDRK4, 511 Fourier points, dt 1e-5) is not
converged: the viscous shock is only about two grid spacings wide. Here: Fourier
pseudo-spectral on [-1, 1) with
integrating-factor RK4, no dealiasing, at two resolutions each in
space and time; the finest solution is written on the released (t, x) grid,
together with the differences that show convergence and the error of the
released reference. Runs in float64 on the GPU like everything else.
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import x64  # noqa: E402,F401  (float64, highest matmul precision)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import scipy.io  # noqa: E402

EXAMPLES = os.path.join(HERE, "..", "..", "external", "jaxpi2", "examples")

PROBLEMS = {
    # lin: symbol of the linear part (Fourier), nonlin(u, k): Fourier transform of the rest
    "burgers": dict(
        data="burgers/data/burgers.mat",
        u0=lambda x: -jnp.sin(jnp.pi * x),
        lin=lambda k: -0.01 / jnp.pi * k**2,
        nonlin=lambda u, k: -0.5j * k * jnp.fft.fft(u**2),
        runs=[(4096, 1e-5), (8192, 1e-5), (8192, 5e-6)],
    ),
}


def solve(p, n, dt, t_out, x_out):
    x = -1.0 + 2.0 * jnp.arange(n) / n
    kint = jnp.fft.fftfreq(n, 1.0 / n)
    k = jnp.pi * kint
    E = jnp.exp(dt * p["lin"](k) / 2)
    E2 = E * E
    N = lambda v: p["nonlin"](jnp.real(jnp.fft.ifft(v)), k)
    per = int(round((t_out[1] - t_out[0]) / dt))
    assert abs(per * dt - (t_out[1] - t_out[0])) < 1e-12

    @jax.jit
    def advance(v):
        def rk4(_, v):
            a = dt * N(v)
            b = dt * N(E * (v + a / 2))
            c = dt * N(E * v + b / 2)
            d = dt * N(E2 * v + E * c)
            return E2 * v + (E2 * a + 2 * E * (b + c) + d) / 6

        return jax.lax.fori_loop(0, per, rk4, v)

    # exact Fourier interpolation onto x_out
    M = jnp.exp(1j * jnp.pi * jnp.outer(kint, jnp.asarray(x_out) + 1.0)) / n

    v = jnp.fft.fft(p["u0"](x))
    out = [jnp.real(v @ M)]
    for _ in range(len(t_out) - 1):
        v = advance(v)
        out.append(jnp.real(v @ M))
    tail = jnp.max(jnp.abs(v[..., jnp.abs(kint) > 0.45 * n])) / jnp.max(jnp.abs(v))  # v may carry a batch axis
    return np.asarray(jnp.stack(out)), float(tail)


def main():
    name, out_dir = sys.argv[1], sys.argv[2]
    p = PROBLEMS[name]
    data = scipy.io.loadmat(os.path.join(EXAMPLES, p["data"]))
    t, x, u_ref = data["t"].ravel(), data["x"].ravel(), data["usol"]

    sols, info = [], {}
    for n, dt in p["runs"]:
        u, tail = solve(p, n, dt, t, x)
        sols.append(u)
        info[f"N{n}_dt{dt:g}_spectral_tail"] = float(tail)
        print(f"{name} N {n} dt {dt:g} done, highest-mode ratio {tail:.1e}", flush=True)

    rel = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))
    u = sols[-1]
    info["space_refinement_diff"] = rel(sols[0], sols[1])
    info["time_refinement_diff"] = rel(sols[1], sols[2])
    info["released_reference_error"] = rel(u_ref, u)
    print(json.dumps(info, indent=1), flush=True)

    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, f"{name}_true.npz"), u=u, t=t, x=x)
    with open(os.path.join(out_dir, f"{name}_true.json"), "w") as f:
        json.dump(dict(runs=p["runs"], **info), f, indent=1)


if __name__ == "__main__":
    main()
