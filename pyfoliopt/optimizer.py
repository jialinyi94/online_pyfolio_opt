import jax
import jax.numpy as jnp
import optax


def mirror_descent(learning_rate, mirror_map, inverse_mirror_map):
    # TODO: implement scheduler for learning rate
    def init_fn(params):
        return optax.EmptyState()
    
    def update_fn(updates, state, params):
        # Apply mirror map to current params
        mirrored_params = jax.tree.map(mirror_map, params)
        
        # Gradient step in mirrored space
        mirrored_update = jax.tree.map(
            lambda m, g: m - learning_rate * g,
            mirrored_params,
            updates
        )
        
        # Map back to parameter space
        new_params = jax.tree.map(inverse_mirror_map, mirrored_update)
        
        return new_params - params, state
    
    return optax.GradientTransformation(init_fn, update_fn)


def egd(learning_rate):
    """Exponentiated Gradient Descent"""
    return mirror_descent(learning_rate, jnp.log, jax.nn.softmax)
