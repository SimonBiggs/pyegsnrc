import pathlib

import jax.numpy as jnp

import pandas as pd
from .. import interpolation

HERE = pathlib.Path(__file__).parent.resolve()

# Data file description in header given as:

# BREMSSTRAHLUNG CROSS-SECTION, TOTAL: (B^2/Z^2)*k*dsig/dk (mb). F. Tessier 2007.
# FORMAT:
#     nT nk; Tvalues; kvalues; "# BREMX.DAT";
#     100 blocks Z=1..100, nk lines of nT values

# For now, just name variables according to header, n_t, n_k, and z.


def get_bremsstrahlung_interpolator():
    raw_data = _load_all_bremsstrahlung_data()

    n_t = jnp.array(raw_data[0].columns.values, dtype=float)
    n_k = jnp.array(raw_data[0].index.values, dtype=float)
    z = jnp.arange(100) + 1

    data = jnp.array(raw_data)
    interpolator = interpolation.create_interpolator((z, n_k, n_t), data)

    return interpolator


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
