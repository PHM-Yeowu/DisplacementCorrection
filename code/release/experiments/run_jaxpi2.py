"""Train a jaxpi2 example unchanged, in float64.

    python run_jaxpi2.py <example> <config> <run_dir> [--config.key=value ...]

The example's own main.py runs from its directory (it reads data/ relatively);
checkpoints and the W&B files go to <run_dir>. Training stops before the first
step if any parameter or optimizer leaf is not float64. With ERROR_TRUE=<npz>
in the environment the log also carries error/true, the relative L2 error against
the converged reference u in that file (make_references.py).
"""

import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import x64  # noqa: E402  (must precede jaxpi)

import numpy as np  # noqa: E402
import jaxpi.training  # noqa: E402
from jaxpi.checkpointing import save_config  # noqa: E402

import pdes  # noqa: E402

_train_loop = jaxpi.training.train_loop


def with_true_error(evaluator, u_true):
    """The example's evaluator, also logging the error against the converged reference as error/true."""

    def evaluate(model, state, *args):
        log_dict = evaluator(model, state, *args)
        log_dict["error/true"] = model.compute_l2_error(state.params, u_true)
        return log_dict

    return evaluate


def train_loop(config, model, *args, **kwargs):
    bad = x64.check_dtypes(model.state)
    if bad:
        raise TypeError("non-float64 leaves: " + ", ".join(bad[:10]))
    print("dtype check: every floating leaf of the train state is float64", flush=True)
    save_config(config, os.environ["WANDB_DIR"])  # the corrections rebuild the backbone from it
    if os.environ.get("ERROR_TRUE"):
        kwargs["evaluator"] = with_true_error(kwargs["evaluator"], np.load(os.environ["ERROR_TRUE"])["u"])
    return _train_loop(config, model, *args, **kwargs)


jaxpi.training.train_loop = train_loop


def main():
    example, config, run_dir = sys.argv[1:4]
    run_dir = os.path.abspath(run_dir)
    os.makedirs(run_dir, exist_ok=True)
    os.environ.setdefault("WANDB_MODE", "offline")
    os.environ["WANDB_DIR"] = run_dir

    exdir = pdes.example_dir(example)
    os.chdir(exdir)
    sys.path.insert(0, exdir)
    sys.argv = [
        "main.py",
        f"--config=configs/{config}.py",
        f"--workdir={run_dir}",
        f"--config.saving.ckpt_path={run_dir}/ckpt",
        *sys.argv[4:],
    ]
    runpy.run_path("main.py", run_name="__main__")


if __name__ == "__main__":
    main()
