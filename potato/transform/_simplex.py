import jax.numpy as jnp
import chex


def fix_share(params: chex.Array, eta):
    uniform_weights = jnp.ones(shape=params) / params.shape[0]
    return (1 - eta) * params + eta * uniform_weights
