from typing import Tuple

from jax import random, jit
import jax.numpy as jnp

from .particles import Particles


def random_walk(
    prng_key: jnp.DeviceArray, particles: Particles, iterations: int,
) -> Tuple[jnp.DeviceArray, Particles]:
    num_particles = particles["position"].shape[-1]

    for _ in range(iterations):
        random_normal_numbers = random.normal(prng_key, shape=(7, num_particles))
        (prng_key,) = random.split(prng_key, 1)

        particles["position"] += random_normal_numbers[0:3, :]
        particles["direction"] += random_normal_numbers[3:6, :]
        particles["energy"] += random_normal_numbers[7, :]

    return prng_key, particles


random_walk = jit(random_walk, static_argnums=(2,))
