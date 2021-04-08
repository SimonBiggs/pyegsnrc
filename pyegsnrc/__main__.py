import jax.numpy as jnp
import matplotlib.pyplot as plt
import typer
from jax import random

from . import lookuptable, particles, randomwalk, timer

random_walk = timer.timer(randomwalk.random_walk)

app = typer.Typer()
main = app


@app.command()
def walk(num_electrons: int, iterations: int, runs: int):
    seed = 0
    prng_key = random.PRNGKey(seed)
    num_electrons_to_plot = 100

    electrons = particles.zeros(num_electrons)

    positions_for_plotting = []
    for _ in range(runs):
        prng_key, electrons = random_walk(prng_key, electrons, iterations)
        positions_for_plotting.append(
            electrons["position"][0:2, 0:num_electrons_to_plot]
        )

    positions_for_plotting = jnp.array(positions_for_plotting)

    plt.plot(
        positions_for_plotting[:, 0, :],
        positions_for_plotting[:, 1, :],
        "o-",
        alpha=0.7,
    )
    plt.axis("equal")
    plt.show()


@app.command()
def lookup():
    lookuptable.main()
