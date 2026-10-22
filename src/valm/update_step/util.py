import jax
from jax import numpy as jnp


def summery_stats(
    values: jax.Array, where: jax.Array | None = None
) -> dict[str, jax.Array]:
    return {
        "mean": jnp.mean(values, where=where),
        "std": jnp.std(values, where=where),
        "min": jnp.min(values, initial=1000, where=where),
        "max": jnp.max(values, initial=-1000, where=where),
    }
