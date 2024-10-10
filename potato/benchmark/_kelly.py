import cvxpy as cp
from cvxpy.error import SolverError
import jax.numpy as jnp
import chex
import logging


def best_kelly_portfolio(
    returns: chex.Array,
    short_selling: bool = False,
    *args, **kwargs
) -> chex.Array:
    r"""Compute the Kelly-criterion optimal portfolio weights for a given set of returns."""
    try:
        return _best_kelly_portfolio(returns, short_selling, *args, **kwargs)
    except SolverError as e:
        logging.warning(f"SolverError: {e}")
        logging.warning("Falling back to 1/N portfolio")
        return jnp.full(returns.shape[1], 1 / returns.shape[1], dtype=returns.dtype, device=returns.device)


def _best_kelly_portfolio(
    returns: chex.Array,
    short_selling: bool = False,
    tol: float = 1e-7,
    *args, **kwargs
):
    returns = jnp.asarray(returns)
    n = returns.shape[1]
    x = cp.Variable(n)
    objective = cp.Maximize(cp.sum(cp.log1p(returns @ x)))
    if not short_selling:
        constraints = [cp.sum(x) == 1, x >= 0]
    else:
        constraints = [cp.sum(cp.abs(x)) <= 1]
    problem = cp.Problem(objective, constraints)
    problem.solve(*args, **kwargs)
    weights = jnp.array(x.value, dtype=returns.dtype, device=returns.device).squeeze()
    # improve numerical stability
    if not short_selling:
        weights = jnp.where(weights < tol, 0.0, weights)
        weights = weights / jnp.sum(weights)
    else:
        weights = jnp.where(jnp.abs(weights) < tol, 0.0, weights)
    return weights
