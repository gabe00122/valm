from typing import cast

import optax
from flax import nnx
from valm.config import (
    AdamWConfig,
    OptimizerConfig,
    SGDConfig,
    WarmupCosineConfig,
)


def make_optimizer(
    model: nnx.Module,
    opt_config: OptimizerConfig,
    total_steps: int,
    gradient_accumulations: int | None,
    wrt: nnx.filterlib.Filter,
) -> nnx.Optimizer:
    schedule_steps = total_steps

    if opt_config.schedule is None:
        tx_lr = opt_config.opt.lr
    elif isinstance(opt_config.schedule, WarmupCosineConfig):
        warmup_steps = int(schedule_steps * opt_config.schedule.warmup_ratio)
        tx_lr = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=opt_config.opt.lr,
            warmup_steps=warmup_steps,
            decay_steps=schedule_steps,
        )

    if isinstance(opt_config.opt, SGDConfig):
        tx = optax.sgd(
            learning_rate=tx_lr,
            momentum=opt_config.opt.momentum,
            nesterov=opt_config.opt.nesterov,
        )
    elif isinstance(opt_config.opt, AdamWConfig):
        tx = optax.adamw(
            learning_rate=tx_lr,
            b1=opt_config.opt.beta1,
            b2=opt_config.opt.beta2,
            weight_decay=opt_config.opt.weight_decay,
            eps=opt_config.opt.eps,
        )

    if opt_config.max_grad_norm is not None:
        tx = optax.chain(optax.clip_by_global_norm(opt_config.max_grad_norm), tx)

    if gradient_accumulations is not None:
        tx = optax.MultiSteps(tx, every_k_schedule=gradient_accumulations)

    return nnx.Optimizer(
        model=model,
        tx=cast(optax.GradientTransformation, tx),
        wrt=wrt,
    )
