import chex
import jax.numpy as jnp


def kelly_loss(
    params: chex.Array,
    returns: chex.Array
) -> chex.Array:
    return -jnp.log(1+jnp.dot(params, returns))
