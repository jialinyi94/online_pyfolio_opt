import jax
import jax.numpy as jnp
import chex

from pyfoliopt.projections import projection_l1_ball


def test_projection_onto_l1_ball():
    # Example usage
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (5,))
    projected_x = projection_l1_ball(x)
    chex.assert_trees_all_equal(jnp.sum(jnp.abs(projected_x)), 1.0)

    # Example with a pytree
    pytree = {'a': jax.random.normal(key, (3,)), 'b': jax.random.normal(key, (2,))}
    projected_pytree = projection_l1_ball(pytree, radius=1.5)
    answer = jax.tree_map(lambda x: jnp.sum(jnp.abs(x)), projected_pytree)
    expected = jax.tree_map(lambda x: 1.5, projected_pytree)
    chex.assert_trees_all_close(answer, expected)

    # Example with radius as a pytree
    radius = {'a': 1.0, 'b': 1.5}
    projected_pytree = projection_l1_ball(pytree, radius=radius)
    answer = jax.tree_map(lambda x: jnp.sum(jnp.abs(x)), projected_pytree)
    chex.assert_trees_all_close(answer, radius)
