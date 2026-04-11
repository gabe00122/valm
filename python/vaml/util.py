import os

import jax
from jax import numpy as jnp
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from zipp import Path


def load_tokenizer(
    tokenizer_path: str | os.PathLike[str] | Path,
) -> PreTrainedTokenizerFast:
    return AutoTokenizer.from_pretrained(tokenizer_path)


def batched_put(target: jax.Array, indices: jax.Array, values: jax.Array) -> jax.Array:
    dnums = jax.lax.ScatterDimensionNumbers(
        update_window_dims=range(2, values.ndim),
        inserted_window_dims=(1,),
        scatter_dims_to_operand_dims=(1,),
        operand_batching_dims=(0,),
        scatter_indices_batching_dims=(0,),
    )

    return jax.lax.scatter(
        target,
        indices[..., None],
        values,
        dnums,
        indices_are_sorted=True,
        unique_indices=True,
    )


def batched_take(target: jax.Array, indices: jax.Array) -> jax.Array:
    slice_sizes = (1, 1) + target.shape[2:]

    dnums = jax.lax.GatherDimensionNumbers(
        offset_dims=tuple(range(2, target.ndim)),
        collapsed_slice_dims=(1,),
        start_index_map=(1,),
        operand_batching_dims=(0,),
        start_indices_batching_dims=(0,),
    )

    return jax.lax.gather(
        target,
        indices[..., None],
        dnums,
        slice_sizes,
        unique_indices=True,
        indices_are_sorted=True,
        mode="promise_in_bounds",
    )


def batched_put_where(
    target: jax.Array, indices: jax.Array, values: jax.Array, where: jax.Array
) -> jax.Array:
    original = batched_take(target, indices)
    values = jnp.where(where, values, original)
    return batched_put(target, indices, values)
