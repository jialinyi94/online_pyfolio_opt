import chex
import jax
import jax.numpy as jnp

from potato.benchmark import best_kelly_portfolio
import chex


def test_kelly_portfolio_basic():
    key = jax.random.PRNGKey(0)
    returns = jax.random.normal(key, (10, 3))
    weights = best_kelly_portfolio(returns)
    chex.assert_shape(weights, (3,))
    chex.assert_trees_all_close(jnp.sum(weights), 1.0, atol=1e-7)
    chex.assert_tree_all_finite(weights)
    assert jnp.all(weights >= 0), "Weights should be non-negative when short selling is not allowed"

    weights_short = best_kelly_portfolio(returns, short_selling=True)
    chex.assert_shape(weights_short, (3,))
    chex.assert_trees_all_close(jnp.sum(jnp.abs(weights_short)), 1.0, atol=1e-5)
    chex.assert_tree_all_finite(weights_short)


def test_kelly_portfolio_short_selling():
    key = jax.random.PRNGKey(0)
    returns = jnp.array(
        [
            [-0.1, -0.2, -0.3],
            [-0.1, -0.2, -0.3],
            [-0.1, -0.2, -0.3],
            [-0.1, -0.2, -0.3],
        ]
    )
    weights = best_kelly_portfolio(returns, short_selling=True)
    expected = jnp.array([0, 0, -1.0], dtype=jnp.float32)
    chex.assert_trees_all_close(weights, expected, atol=1e-5)
