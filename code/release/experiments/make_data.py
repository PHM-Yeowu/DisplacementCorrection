"""Reference data of the Burgers benchmark with the viscosity 0.003 / pi (examples/burgers_nu3).

    python make_data.py burgers_nu3 <out.mat>

The jaxpi2 burgers problem with a smaller viscosity, on the grid of the jaxpi2 burgers data, solved as in
make_references.py with more Fourier modes for the narrower shock, at two resolutions in space and two time steps.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import make_references as mr  # noqa: E402  (imports x64)

import numpy as np  # noqa: E402
import scipy.io  # noqa: E402

BURGERS = {"burgers_nu3": dict(nu=0.003 / np.pi, runs=[(8192, 1e-5), (16384, 1e-5), (16384, 5e-6)])}


def burgers(nu, n, dt, t, x):
    p = dict(mr.PROBLEMS["burgers"], lin=lambda k: -nu * k**2)
    return mr.solve(p, n, dt, t, x)[0]


def main():
    name, out = sys.argv[1], sys.argv[2]
    ref = scipy.io.loadmat(os.path.join(mr.EXAMPLES, "burgers", "data", "burgers.mat"))
    t, x = ref["t"].ravel(), ref["x"].ravel()
    b = BURGERS[name]
    sols = [burgers(b["nu"], n, dt, t, x) for n, dt in b["runs"]]
    u = sols[-1]
    rel = lambda a, c: float(np.linalg.norm(a - c) / np.linalg.norm(c))
    print(f"{name}: space {rel(sols[0], sols[1]):.2e}  time {rel(sols[1], sols[2]):.2e}", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    scipy.io.savemat(out, {"usol": u, "t": t[:, None], "x": x[:, None]})
    print("wrote", out, u.shape, flush=True)


if __name__ == "__main__":
    main()
