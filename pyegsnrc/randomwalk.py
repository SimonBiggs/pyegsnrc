from typing import Tuple

import jax.numpy as jnp
from jax import jit, random

from . import particles


def random_walk(
    prng_key: jnp.DeviceArray, electrons: particles.Type, iterations: int
) -> Tuple[jnp.DeviceArray, particles.Type]:
    num_electrons = electrons["position"].shape[-1]

    for _ in range(iterations):
        random_normal_numbers = random.normal(prng_key, shape=(7, num_electrons))
        (prng_key,) = random.split(prng_key, 1)

        electrons["position"] += random_normal_numbers[0:3, :]
        electrons["direction"] += random_normal_numbers[3:6, :]
        electrons["energy"] += random_normal_numbers[7, :]

    return prng_key, electrons


random_walk = jit(random_walk, static_argnums=(2,))
