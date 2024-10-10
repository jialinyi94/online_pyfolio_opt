from potato.loss import kelly_loss

import chex
import jax.numpy as jnp


def test_kelly_loss_1d_returns():
    params = jnp.array([0.5, 0.5])
    returns = jnp.array([0.1, 0.2])
    chex.assert_trees_all_close(
        kelly_loss(params, returns),
        -jnp.log(1 + jnp.dot(params, returns))
    )


def test_kelly_loss_2d_returns():
    params = jnp.array([0.5, 0.5])
    returns = jnp.array([[0.1, 0.2], [0.3, 0.4]])
    chex.assert_trees_all_close(
        kelly_loss(params, returns),
        jnp.mean(-jnp.log(1 + jnp.dot(params, returns.T)))
    )