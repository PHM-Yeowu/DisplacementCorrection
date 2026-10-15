"""Train one correction stage on a jaxpi2 backbone, in float64.

    python run_correction.py --example kdv --backbone runs/backbones/kdv_s42 \
        --out runs/corrections/kdv_s42_disp1 [--prev <stage dir> ...] \
        [--form disp|add] [--steps N] [--seed S]

The backbone is restored from its checkpoint; earlier stages (--prev, in order)
are applied on top of it. The corrector has the architecture of the backbone.
The stage writes its parameters and scales (stage.npz, stage.json) and the
errors before and after it (result.json) to --out.
"""

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import x64  # noqa: E402  (must precede jaxpi)

import numpy as np  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax import random, vmap  # noqa: E402
from jax.flatten_util import ravel_pytree  # noqa: E402

from jaxpi.checkpointing import create_checkpoint_manager, get_ckpt_path, restore_checkpoint  # noqa: E402
from jaxpi.models import create_arch, create_model  # noqa: E402
from jaxpi.training import train  # noqa: E402

import pdes  # noqa: E402
from correction import Correction, CorrectionEvaluator, PoolSampler, corrected, jets, prepare  # noqa: E402


def load_backbone(example, run_dir):
    config = pdes.load_config(run_dir)
    model, u_ref, t_star, x_star = pdes.build(example, config)
    ckpt_path = get_ckpt_path(config)
    if config.training.get("num_time_windows", 1) > 1:
        ckpt_path = os.path.join(ckpt_path, "time_window_1")
    state = restore_checkpoint(create_checkpoint_manager(config.saving, ckpt_path), model.state)
    return config, model, state.params, u_ref, t_star, x_star


def load_stage(stage_dir):
    with open(os.path.join(stage_dir, "stage.json")) as f:
        meta = json.load(f)
    arch_cfg = x64_config(meta["arch"])
    arch = create_arch(arch_cfg)
    template = arch.init(random.PRNGKey(0), jnp.ones(2))
    _, unravel = ravel_pytree(template)
    params = unravel(jnp.asarray(np.load(os.path.join(stage_dir, "stage.npz"))["flat"]))
    return arch.apply, params, meta


def x64_config(d):
    import ml_collections

    return ml_collections.ConfigDict(pdes._tuples(d))


def corrector_config(bb, args, out):
    config = x64_config(bb.to_dict())
    config.arch.out_dim = 2 if args.form == "disp" else 1

    config.correction = x64_config({"form": args.form})
    config.pseudo_time.enabled = False
    config.pseudo_time.strategy = "constant"
    config.causal.enabled = False
    config.loss_weighting.loss_weights = x64_config({"ics": 1.0, "res": 1.0})
    config.training.max_steps = args.steps
    config.training.batch_size = args.batch or bb.training.batch_size
    config.training.resume = False
    config.logging.log_every_steps = args.log_every
    config.logging.log_causal_weights = False
    config.saving.ckpt_path = out
    config.saving.save_every_steps = 10000
    config.wandb.project = "DISPLACEMENT-CORRECTION"
    config.wandb.name = os.path.basename(out.rstrip("/"))
    config.seed = args.seed
    return config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--example", required=True)
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prev", nargs="*", default=[])
    ap.add_argument("--form", choices=["disp", "add"], default="disp")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=0)
    ap.add_argument("--pool", type=int, default=2**20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=100)
    args = ap.parse_args()

    out = os.path.abspath(args.out)
    backbone = os.path.abspath(args.backbone)
    prev = [os.path.abspath(p) for p in args.prev]
    os.makedirs(out, exist_ok=True)
    os.environ.setdefault("WANDB_MODE", "offline")
    os.environ["WANDB_DIR"] = out
    pdes.enter(args.example)
    t_start = time.time()

    bb, model, params, u_ref, t_star, x_star = load_backbone(args.example, backbone)
    U = lambda t, x: model.neural_net(params, t, x)
    m, R = pdes.residual(args.example, bb)

    # R on the backbone's jet must reproduce the example's r_net
    tt = jnp.linspace(t_star[0], t_star[-1], 64)
    xx = jnp.linspace(x_star[0], x_star[-1], 64)
    r_net = vmap(model.r_net, (None, 0, 0))(params, tt, xx)
    r_jet = vmap(lambda a, b: R(jets(U, m)(a, b)))(tt, xx)
    mismatch = float(jnp.max(jnp.abs(r_net - r_jet)) / jnp.sqrt(jnp.mean(r_net**2)))
    print(f"R vs r_net: max relative difference {mismatch:.2e}", flush=True)
    assert mismatch < 1e-9

    for stage_dir in prev:
        apply_fn, p, meta = load_stage(stage_dir)
        U = corrected(U, apply_fn, p, meta["s"], meta["sigma"], meta["form"] == "disp")

    data = prepare(U, R, m, u_ref, t_star, x_star, args.pool, random.PRNGKey(1000 + len(prev)))
    print("s {:.3e}  sigma {:.3e}  rms r {:.3e}  rms e0 {:.3e}".format(
        float(data["s"]), float(data["sigma"]), float(data["r_rms"]),
        float(jnp.sqrt(jnp.mean(data["ic"]["e0"] ** 2)))), flush=True)
    t_prep = time.time() - t_start

    config = corrector_config(bb, args, out)
    # zero output layer: the stage starts from U itself (E = 0)
    params0 = create_arch(config.arch).init(random.PRNGKey(config.seed), jnp.ones(config.input_dim))
    top = dict(params0["params"])
    last = max((k for k in top if k.startswith("Dense_")), key=lambda k: int(k.split("_")[1]))
    top[last] = jax.tree.map(jnp.zeros_like, top[last])
    params0 = {"params": top}
    refs = {"ref": u_ref}
    true_path = os.path.join(HERE, "..", "..", "..", "runs", "references", f"{args.example}_true.npz")
    if os.path.exists(true_path):  # converged solution where the released one is not (make_references.py)
        refs["true"] = jnp.asarray(np.load(true_path)["u"])
    corr = create_model(config, Correction, R=R, m=m, data=data, refs=refs, params=params0)
    bad = x64.check_dtypes(corr.state)
    assert not bad, bad
    before = {k: float(v) for k, v in corr.errors(corr.state.params).items()}
    print("backbone:", before, flush=True)

    sampler = PoolSampler(data["pool"], config.training.batch_size, seed=args.seed)
    t_train = time.time()
    corr = train(config, corr, sampler, evaluator=CorrectionEvaluator(config))
    t_train = time.time() - t_train

    params_c = corr.state.params
    after = {k: float(v) for k, v in corr.errors(params_c).items()}
    print("corrected:", after, flush=True)

    flat, _ = ravel_pytree(params_c)
    np.savez(os.path.join(out, "stage.npz"), flat=np.asarray(flat))
    meta = dict(example=args.example, backbone=backbone, prev=prev, form=args.form,
                arch=config.arch.to_dict(), s=float(data["s"]), sigma=float(data["sigma"]),
                steps=args.steps, batch=config.training.batch_size, seed=args.seed)
    with open(os.path.join(out, "stage.json"), "w") as f:
        json.dump(meta, f, indent=1)
    result = dict(meta, before=before, after=after, r_rms=float(data["r_rms"]),
                  time_prepare=t_prep, time_train=t_train)
    with open(os.path.join(out, "result.json"), "w") as f:
        json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
