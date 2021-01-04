# RegularGridInterpolator

# https://github.com/scipy/scipy/blob/ab1c0907fe9255582397db04592d6066745018d3/scipy/interpolate/interpolate.py#L2338-L2564

import itertools
from functools import partial

import jax.numpy as jnp
from jax import jit


def create_interpolator(points, values):
    if not hasattr(values, "ndim"):
        # allow reasonable duck-typed values
        values = jnp.asarray(values)

    if len(points) > values.ndim:
        raise ValueError(
            "There are %d point arrays, but values has %d "
            "dimensions" % (len(points), values.ndim)
        )

    if hasattr(values, "dtype") and hasattr(values, "astype"):
        if not jnp.issubdtype(values.dtype, jnp.inexact):
            values = values.astype(float)

    for i, p in enumerate(points):
        if not jnp.all(jnp.diff(p) > 0.0):
            raise ValueError(
                "The points in dimension %d must be strictly ascending" % i
            )
        if not jnp.asarray(p).ndim == 1:
            raise ValueError("The points in dimension %d must be 1-dimensional" % i)
        if not values.shape[i] == len(p):
            raise ValueError(
                "There are %d points and %d values in "
                "dimension %d" % (len(p), values.shape[i], i)
            )
    grid = tuple([jnp.asarray(p) for p in points])
    ndim = len(grid)

    def interpolator(xi, method="linear"):
        if method not in ["linear", "nearest"]:
            raise ValueError("Method '%s' is not defined" % method)

        xi = _ndim_coords_from_arrays(xi, ndim)
        if xi.shape[-1] != len(grid):
            raise ValueError(
                "The requested sample points xi have dimension "
                "%d, but this RegularGridInterpolator has "
                "dimension %d" % (xi.shape[1], ndim)
            )

        xi_shape = xi.shape
        xi = xi.reshape(-1, xi_shape[-1])

        for i, p in enumerate(xi.T):
            if not jnp.logical_and(jnp.all(grid[i][0] <= p), jnp.all(p <= grid[i][-1])):
                raise ValueError(
                    "One of the requested xi is out of bounds in dimension %d" % i
                )

        indices, norm_distances = _find_indices(xi.T, grid)
        if method == "linear":
            result = _evaluate_linear(values, indices, norm_distances)
        elif method == "nearest":
            result = _evaluate_nearest(values, indices, norm_distances)

        return result.reshape(xi_shape[:-1] + values.shape[ndim:])

    return interpolator


@jit
def _evaluate_linear(values, indices, norm_distances):
    # slice for broadcasting over trailing dimensions in self.values
    vslice = (slice(None),) + (None,) * (values.ndim - len(indices))

    # find relevant values
    # each i and i+1 represents a edge
    edges = itertools.product(*[[i, i + 1] for i in indices])
    collated_values = 0.0
    for edge_indices in edges:
        weight = 1.0
        for ei, i, yi in zip(edge_indices, indices, norm_distances):
            weight *= jnp.where(ei == i, 1 - yi, yi)
        collated_values += jnp.asarray(values[edge_indices]) * weight[vslice]
    return collated_values


@jit
def _evaluate_nearest(values, indices, norm_distances):
    idx_res = [jnp.where(yi <= 0.5, i, i + 1) for i, yi in zip(indices, norm_distances)]
    return values[tuple(idx_res)]


@partial(jit, static_argnums=(1,))
def _find_indices(xi, grid):
    # find relevant edges between which xi are situated
    indices = []
    # compute distance to lower edge in unity units
    norm_distances = []
    # iterate through dimensions
    for x, grid in zip(xi, grid):
        i = jnp.searchsorted(grid, x) - 1
        indices.append(i)
        norm_distances.append((x - grid[i]) / (grid[i + 1] - grid[i]))
    return indices, norm_distances


@partial(jit, static_argnums=(1,))
def _ndim_coords_from_arrays(points, ndim):
    """
    Convert a tuple of coordinate arrays to a (..., ndim)-shaped array.
    """
    if isinstance(points, tuple) and len(points) == 1:
        # handle argument tuple
        points = points[0]
    if isinstance(points, tuple):
        p = jnp.broadcast_arrays(*points)
        n = len(p)
        for j in range(1, n):
            if p[j].shape != p[0].shape:
                raise ValueError("coordinate arrays do not have the same shape")
        points = jnp.empty(p[0].shape + (len(points),), dtype=float)
        for j, item in enumerate(p):
            points[..., j] = item
    else:
        points = jnp.asarray(points)
        if points.ndim == 1:
            if ndim is None:
                points = points.reshape(-1, 1)
            else:
                points = points.reshape(-1, ndim)
    return points
