import jax
import jax.numpy as jnp
import optax
import optax._src.base


def mirror_descent(
    lr_schedule: optax._src.base.ScalarOrSchedule,
    mirror_map,
    inverse_mirror_map
):
    def init_fn(params):
        return optax.ScaleByScheduleState(count=jnp.zeros([], jnp.int32))
    
    def update_fn(updates, state, params):
        learning_rate = lr_schedule(state.count) if callable(lr_schedule) else lr_schedule
        new_state = optax.ScaleByScheduleState(count=state.count + 1)
        
        # Apply mirror map to current params
        mirrored_params = jax.tree_map(mirror_map, params)
        
        # Gradient step in mirrored space
        mirrored_update = jax.tree_map(
            lambda m, g: m - learning_rate * g,
            mirrored_params,
            updates
        )
        
        # Map back to parameter space
        new_params = jax.tree_map(inverse_mirror_map, mirrored_update)
        
        return new_params - params, new_state
    
    return optax.GradientTransformation(init_fn, update_fn)


def egd(learning_rate: optax._src.base.ScalarOrSchedule):
    """Exponentiated Gradient Descent"""
    return mirror_descent(learning_rate, jnp.log, jax.nn.softmax)
