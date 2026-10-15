"""Errors of a trained jaxpi2 run (a backbone or its continued training) against the released and the converged
reference.

    python eval_backbone.py <example> <run dir> [<run dir> ...]

Writes the relative L2 errors to <run dir>/eval.json.
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import run_correction as rc  # noqa: E402  (imports x64 before jaxpi)

import numpy as np  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax import jit, vmap  # noqa: E402


def evaluate(example, run_dir, chunk=8192):
    _, model, params, u_ref, t_star, x_star = rc.load_backbone(example, run_dir)
    fn = jit(vmap(lambda t, x: model.neural_net(params, t, x)))
    TT, XX = jnp.meshgrid(t_star, x_star, indexing="ij")
    tt, xx = TT.ravel(), XX.ravel()
    U = np.concatenate([np.asarray(fn(tt[i:i + chunk], xx[i:i + chunk])) for i in range(0, tt.size, chunk)])
    U = U.reshape(TT.shape)
    refs = {"ref": np.asarray(u_ref)}
    true = os.path.join(HERE, "..", "..", "..", "runs", "references", f"{example}_true.npz")
    if os.path.exists(true):
        refs["true"] = np.load(true)["u"]
    out = {f"error/{k}": float(np.linalg.norm(U - u) / np.linalg.norm(u)) for k, u in refs.items()}
    with open(os.path.join(run_dir, "eval.json"), "w") as f:
        json.dump(dict(example=example, run=run_dir, **out), f, indent=1)
    print(run_dir, out, flush=True)


if __name__ == "__main__":  # runs of one example per call (pdes.enter changes directory and imports)
    example = sys.argv[1]
    dirs = [os.path.abspath(d) for d in sys.argv[2:]]
    rc.pdes.enter(example)
    for d in dirs:
        evaluate(example, d)
