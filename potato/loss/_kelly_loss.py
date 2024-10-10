import chex
import jax.numpy as jnp


def kelly_loss(
    params: chex.Array,
    returns: chex.Array
) -> chex.Array:
    if params.ndim != 1:
        raise ValueError("params must be 1D")
    if returns.ndim == 1:
        return -jnp.log(1+jnp.dot(params, returns))
    elif returns.ndim == 2:
        return -jnp.mean(jnp.log(1+jnp.dot(params, returns.T)))
    else:
        raise ValueError("returns must be 1D or 2D")

