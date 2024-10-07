import chex
import jax.numpy as jnp


def neg_logdot_loss(relative_prices: chex.Array, portfolio: chex.Array):
    return -jnp.log(jnp.dot(relative_prices, portfolio))
