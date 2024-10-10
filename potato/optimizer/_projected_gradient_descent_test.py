from potato.optimizer import proj_gd
from potato.projection import projection_l1_ball

import jax
import jax.numpy as jnp
import chex
import optax


def test_proj_gd_basic():
    # Define a simple learning rate schedule
    lr_schedule = 0.1

    # Define a simple projection function (e.g., clipping to [0, 1])
    def projection_fn(x):
        return jnp.clip(x, 0, 1)

    # Initialize the optimizer
    optimizer = proj_gd(lr_schedule, projection_fn)
    params = jnp.array([0.5, 1.5, -0.5])
    state = optimizer.init(params)

    # Define some dummy gradients
    updates = jnp.array([0.1, -0.2, 0.3])

    # Perform an update step
    transformed_updates, new_state = optimizer.update(updates, state, params)

    # Check the new state
    assert new_state.count == 1

    # Check the transformed updates
    expected_params = params - lr_schedule * updates
    expected_params_proj = jnp.clip(expected_params, 0, 1)
    expected_transformed_updates = expected_params_proj - params

    chex.assert_trees_all_close(transformed_updates, expected_transformed_updates)


def test_proj_gd_numeric():
    def objective_fn(x):
        return (x[0]+1)**2 + (x[1]-1)**2
    
    optimizer = proj_gd(1e-1, projection_fn=projection_l1_ball)
    opt_state = optimizer.init(jnp.array([0.0, 0.0]))

    params = jnp.array([0.0, 0.0])
    loss_history = []
    for _ in range(100):
        loss, grad = jax.value_and_grad(objective_fn)(params)
        updates, opt_state = optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        loss_history.append(loss)
    
    chex.assert_trees_all_close(params, jnp.array([-0.5, 0.5]))


def test_proj_gd_l1_ball():
    # Define a simple learning rate schedule
    lr_schedule = 0.1

    # Define a simple projection function (e.g., projection onto the L1 ball)
    def projection_fn(x):
        return projection_l1_ball(x, radius=1.0)

    # Initialize the optimizer
    optimizer = proj_gd(lr_schedule, projection_fn)
    params = jnp.array([0.5, 1.5, -0.5])
    state = optimizer.init(params)

    # Define some dummy gradients
    updates = jnp.array([0.1, -0.2, 0.3])

    # Perform an update step
    transformed_updates, new_state = optimizer.update(updates, state, params)

    # Check the new state
    assert new_state.count == 1

    # Check the transformed updates
    expected_params = params - lr_schedule * updates
    expected_params_proj = projection_l1_ball(expected_params, radius=1.0)
    expected_transformed_updates = expected_params_proj - params

    chex.assert_trees_all_close(transformed_updates, expected_transformed_updates)


def test_proj_gd_pytree_inputs():
    # Define a simple learning rate schedule
    lr_schedule = 0.1

    # Define a simple projection function (e.g., clipping to [0, 1])
    def projection_fn(x):
        return jnp.clip(x, 0, 1)

    # Initialize the optimizer
    optimizer = proj_gd(lr_schedule, projection_fn)
    params = {'a': jnp.array([0.5, 1.5, -0.5]), 'b': jnp.array([-0.5, 1.5])}
    state = optimizer.init(params)

    # Define some dummy gradients
    updates = {'a': jnp.array([0.1, -0.2, 0.3]), 'b': jnp.array([-0.1, 0.2])}

    # Perform an update step
    transformed_updates, new_state = optimizer.update(updates, state, params)

    # Check the new state
    assert new_state.count == 1

    # Check the transformed updates
    expected_params = jax.tree.map(lambda p, g: p - lr_schedule * g, params, updates)
    expected_params_proj = jax.tree.map(lambda x: jnp.clip(x, 0, 1), expected_params)
    expected_transformed_updates = jax.tree.map(lambda p_proj, p: p_proj - p, expected_params_proj, params)

    chex.assert_trees_all_close(transformed_updates, expected_transformed_updates)