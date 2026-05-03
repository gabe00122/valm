from typing import Any, cast

import jax
from flax import nnx
from jax import numpy as jnp


def load_param(target: nnx.Param[jax.Array] | None, value):
    assert target is not None
    value = jnp.asarray(value, device=target.device)
    assert value.shape == target.shape
    assert value.dtype == target.dtype
    target[...] = value


def wrap_param(node: nnx.Module, param):
    for path, value in nnx.iter_graph(node):
        if isinstance(value, nnx.Param):
            *path, key = path

            target: Any = node
            for p in path:
                if isinstance(p, int):
                    target = target[p]
                else:
                    target = getattr(target, cast(str, p))

            setattr(target, cast(str, key), param(value[...]))
