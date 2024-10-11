import jax
import jax.numpy as jnp
import chex


def fix_share(params, eta):
    def _fix_share(_params: chex.Array):
        uniform_weights = jnp.ones(shape=_params) / _params.shape[0]
        return (1 - eta) * _params + eta * uniform_weights
    
    return jax.tree_map(_fix_share, params)
