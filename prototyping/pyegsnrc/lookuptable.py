from jax import random
import jax.numpy as jnp

from .data import load
from . import timer


def main():
    interpolator = load.get_bremsstrahlung_interpolator()

    seed = 0
    input_prng_key = random.PRNGKey(seed)
    num_items = int(1e6)
    iterations = 10
    runs = 10

    iterative_sum = jnp.zeros((num_items,))

    for _ in range(runs):
        _iterations(input_prng_key, iterative_sum, interpolator, iterations)


@timer.timer
def _iterations(input_prng_key, iterative_sum, interpolator, iterations):
    prng_keys = random.split(input_prng_key, 4)
    num_items = len(iterative_sum)

    random_numbers_shape = (num_items, iterations)
    z = random.uniform(prng_keys[0], shape=random_numbers_shape, minval=1, maxval=100)

    packed_n_k = random.uniform(
        prng_keys[1], random_numbers_shape, minval=-15, maxval=15
    )
    n_k = load.unpack_n_k(packed_n_k)

    packed_n_t = random.uniform(prng_keys[2], random_numbers_shape, minval=-6, maxval=6)
    n_t = load.unpack_n_t(packed_n_t)

    for i in range(iterations):
        iterative_sum += interpolator(z[:, i], n_k[:, i], n_t[:, i])

    return prng_keys[3], iterative_sum
