import chex
import jax.numpy as jnp


def neg_logdot_loss(
    params: chex.Array,
    relative_prices: chex.Array
):
    return -jnp.log(jnp.dot(params, relative_prices))
