import pathlib

import pandas as pd


HERE = pathlib.Path(__file__).parent.resolve()


def load_bremsstrahlung(z):
    directory = HERE.joinpath("bremsstrahlung")
    filename = directory.joinpath(f"Z{str(z).zfill(3)}.csv")

    return pd.read_csv(filename)
