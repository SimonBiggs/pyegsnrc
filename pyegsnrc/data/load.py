import pathlib

import jax.numpy as jnp
import pandas as pd
from jax import jit, ops

from .. import interpolation

HERE = pathlib.Path(__file__).parent.resolve()

# Data file description in header given as:

# BREMSSTRAHLUNG CROSS-SECTION, TOTAL: (B^2/Z^2)*k*dsig/dk (mb). F. Tessier 2007.
# FORMAT:
#     nT nk; Tvalues; kvalues; "# BREMX.DAT";
#     100 blocks Z=1..100, nk lines of nT values

# For now, just name variables according to header, n_t, n_k, and z.

PACKED_N_K_BOUNDS = 100


def get_bremsstrahlung_interpolator():
    raw_data = _load_all_bremsstrahlung_data()

    n_t = jnp.array(raw_data[0].columns.values, dtype=float)
    n_k = jnp.array(raw_data[0].index.values, dtype=float)
    z = jnp.arange(100) + 1

    data = jnp.array(raw_data)

    packed_n_k = _pack_n_k_for_interp(n_k)
    packed_n_t = _pack_n_t_for_interp(n_t)

    _interpolator = interpolation.create_interpolator((z, packed_n_k, packed_n_t), data)

    def interpolator(z, n_k, n_t):
        packed_n_k = _pack_n_k_for_interp(n_k)
        packed_n_t = _pack_n_t_for_interp(n_t)

        xi = jnp.vstack([z, packed_n_k, packed_n_t]).T

        return _interpolator(xi)

    return interpolator


def _pack_n_k_for_interp(n_k):
    packed_n_k = jnp.log(n_k / (1 - n_k))
    packed_n_k = ops.index_update(
        packed_n_k, ops.index[packed_n_k < -PACKED_N_K_BOUNDS], -PACKED_N_K_BOUNDS
    )
    packed_n_k = ops.index_update(
        packed_n_k, ops.index[packed_n_k > PACKED_N_K_BOUNDS], PACKED_N_K_BOUNDS
    )
    return packed_n_k


def unpack_n_k(packed_n_k):
    n_k = jnp.exp(packed_n_k) / (1 + jnp.exp(packed_n_k))
    return n_k


def _pack_n_t_for_interp(n_t):
    packed_n_t = jnp.log(n_t)
    return packed_n_t


def unpack_n_t(packed_n_t):
    n_t = jnp.exp(packed_n_t)
    return n_t


def get_raw_bremsstrahlung_data():
    raw_data = _load_all_bremsstrahlung_data()

    n_t = jnp.array(raw_data[0].columns.values, dtype=float)
    n_k = jnp.array(raw_data[0].index.values, dtype=float)
    z = jnp.arange(100) + 1

    data = jnp.array(raw_data)

    return (z, n_k, n_t), data


def _load_all_bremsstrahlung_data():
    data = []
    for i in range(100):
        z = i + 1

        data.append(_load_single_bremsstrahlung_z_file(z))

    return data


def _load_single_bremsstrahlung_z_file(z):
    directory = HERE.joinpath("bremsstrahlung")
    filename = directory.joinpath(f"Z{str(z).zfill(3)}.csv")

    return pd.read_csv(filename, index_col=0)
