import jax
import jax.flatten_util
import jax.numpy as jnp
from typing import Any
import chex


def projection_l1_ball(pytree: Any, radius: chex.Scalar | Any = 1.0) -> Any:
    r"""Projection onto the L1 ball.

    .. math::

        \underset{p}{\text{argmin}} ~ ||x - p||_2^2 \quad \textrm{subject to} \quad
        ||p||_1 \le \text{radius}

    where :math:`x` is the input pytree.

    By default, we project to the unit L1 ball (`radius=1.0`).

    Args:
        pytree: pytree to project.
        radius: radius of the L1 ball, a scalar or a pytree (default: 1.0).
    Returns:
        projected pytree, with the same structure as ``pytree``.
    """
    
    def project_array(x, r):
        # Sort |x| in descending order
        sorted_abs_x = jnp.sort(jnp.abs(x))[::-1]
        
        # Compute the cumulative sum
        cumsum = jnp.cumsum(sorted_abs_x)
        
        # Find the number of elements to keep
        k = jnp.sum(sorted_abs_x > (cumsum - r) / jnp.arange(1, len(x) + 1))
        
        # Compute the threshold
        theta = jnp.where(k > 0, (cumsum[k - 1] - r) / k, 0.0)
        
        # Return the projected vector
        return jnp.sign(x) * jnp.maximum(jnp.abs(x) - theta, 0)
    
    # Use jax.tree_util.tree_map to apply the projection to each leaf of the pytree
    if isinstance(radius, chex.Scalar):
        new_pytree = jax.tree.map(lambda x: project_array(x, radius), pytree)
    else:
        new_pytree = jax.tree.map(project_array, pytree, radius)
    return new_pytree
