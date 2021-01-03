# RegularGridInterpolator

# https://github.com/scipy/scipy/blob/ab1c0907fe9255582397db04592d6066745018d3/scipy/interpolate/interpolate.py#L2338-L2564

import itertools

from jax import jit
import jax.numpy as jnp
from jax.ops import index, index_update


class RegularGridInterpolator(object):
    def __init__(
        self, points, values, method="linear", bounds_error=True, fill_value=jnp.nan
    ):
        if method not in ["linear", "nearest"]:
            raise ValueError("Method '%s' is not defined" % method)
        self.method = method
        self.bounds_error = bounds_error

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

        self.fill_value = fill_value
        if fill_value is not None:
            fill_value_dtype = jnp.asarray(fill_value).dtype
            if hasattr(values, "dtype") and not jnp.can_cast(
                fill_value_dtype, values.dtype, casting="same_kind"
            ):
                raise ValueError(
                    "fill_value must be either 'None' or "
                    "of a type compatible with values"
                )

        for i, p in enumerate(points):
            if not jnp.all(jnp.diff(p) > 0.0):
                raise ValueError(
                    "The points in dimension %d must be strictly " "ascending" % i
                )
            if not jnp.asarray(p).ndim == 1:
                raise ValueError(
                    "The points in dimension %d must be " "1-dimensional" % i
                )
            if not values.shape[i] == len(p):
                raise ValueError(
                    "There are %d points and %d values in "
                    "dimension %d" % (len(p), values.shape[i], i)
                )
        self.grid = tuple([jnp.asarray(p) for p in points])
        self.values = values

    def __call__(self, xi, method=None):
        """
        Interpolation at coordinates
        Parameters
        ----------
        xi : ndarray of shape (..., ndim)
            The coordinates to sample the gridded data at
        method : str
            The method of interpolation to perform. Supported are "linear" and
            "nearest".
        """
        method = self.method if method is None else method
        if method not in ["linear", "nearest"]:
            raise ValueError("Method '%s' is not defined" % method)

        ndim = len(self.grid)
        xi = _ndim_coords_from_arrays(xi, ndim)
        if xi.shape[-1] != len(self.grid):
            raise ValueError(
                "The requested sample points xi have dimension "
                "%d, but this RegularGridInterpolator has "
                "dimension %d" % (xi.shape[1], ndim)
            )

        xi_shape = xi.shape
        xi = xi.reshape(-1, xi_shape[-1])

        if self.bounds_error:
            for i, p in enumerate(xi.T):
                if not jnp.logical_and(
                    jnp.all(self.grid[i][0] <= p), jnp.all(p <= self.grid[i][-1])
                ):
                    raise ValueError(
                        "One of the requested xi is out of bounds "
                        "in dimension %d" % i
                    )

        indices, norm_distances, out_of_bounds = self._find_indices(xi.T)
        if method == "linear":
            result = self._evaluate_linear(indices, norm_distances, out_of_bounds)
        elif method == "nearest":
            result = self._evaluate_nearest(indices, norm_distances, out_of_bounds)
        if not self.bounds_error and self.fill_value is not None:
            result[out_of_bounds] = self.fill_value

        return result.reshape(xi_shape[:-1] + self.values.shape[ndim:])

    def _evaluate_linear(self, indices, norm_distances, out_of_bounds):
        # slice for broadcasting over trailing dimensions in self.values
        vslice = (slice(None),) + (None,) * (self.values.ndim - len(indices))

        # find relevant values
        # each i and i+1 represents a edge
        edges = itertools.product(*[[i, i + 1] for i in indices])
        values = 0.0
        for edge_indices in edges:
            weight = 1.0
            for ei, i, yi in zip(edge_indices, indices, norm_distances):
                weight *= jnp.where(ei == i, 1 - yi, yi)
            values += jnp.asarray(self.values[edge_indices]) * weight[vslice]
        return values

    def _evaluate_nearest(self, indices, norm_distances, out_of_bounds):
        idx_res = [
            jnp.where(yi <= 0.5, i, i + 1) for i, yi in zip(indices, norm_distances)
        ]
        return self.values[tuple(idx_res)]

    def _find_indices(self, xi):
        # find relevant edges between which xi are situated
        indices = []
        # compute distance to lower edge in unity units
        norm_distances = []
        # check for out of bounds xi
        out_of_bounds = jnp.zeros((xi.shape[1]), dtype=bool)
        # iterate through dimensions
        for x, grid in zip(xi, self.grid):
            i = jnp.searchsorted(grid, x) - 1
            i = index_update(i, index[i < 0], 0)
            i = index_update(i, index[i > grid.size - 2], grid.size - 2)
            indices.append(i)
            norm_distances.append((x - grid[i]) / (grid[i + 1] - grid[i]))
            if not self.bounds_error:
                out_of_bounds += x < grid[0]
                out_of_bounds += x > grid[-1]
        return indices, norm_distances, out_of_bounds


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


_ndim_coords_from_arrays = jit(_ndim_coords_from_arrays, static_argnums=(1,))
