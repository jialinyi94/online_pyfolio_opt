import jax
import jax.numpy as jnp
import optax
import optax._src.base


def proj_gd(
    lr_schedule: optax._src.base.ScalarOrSchedule,
    projection_fn,
):
    def init_fn(params):
        return optax.ScaleByScheduleState(count=jnp.zeros([], jnp.int32))
    
    def update_fn(updates, state, params):
        learning_rate = lr_schedule(state.count) if callable(lr_schedule) else lr_schedule
        new_state = optax.ScaleByScheduleState(count=state.count + 1)
        
        # Projected gradient step
        new_params = jax.tree.map(
            lambda p, g: projection_fn(p - learning_rate * g),
            params,
            updates
        )

        transformed_updates = jax.tree_map(
            lambda p_proj, p: p_proj - p, 
            new_params, params
        )
        
        return transformed_updates, new_state
    
    return optax.GradientTransformation(init_fn, update_fn)