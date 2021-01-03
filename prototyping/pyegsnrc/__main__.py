import matplotlib.pyplot as plt

from jax import random
import jax.numpy as jnp


from .particles import Particles
from .timer import timer
from .randomwalk import random_walk

random_walk = timer(random_walk)


def main():
    seed = 0
    prng_key = random.PRNGKey(seed)
    num_particles = int(1e6)
    iterations = 10
    runs = 10
    num_particles_to_plot = 100

    particles = Particles.zeros(num_particles)

    positions_for_plotting = []
    for _ in range(runs):
        prng_key, particles = random_walk(prng_key, particles, iterations)
        positions_for_plotting.append(
            particles["position"][0:2, 0:num_particles_to_plot]
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
