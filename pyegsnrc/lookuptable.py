import jax.numpy as jnp
from jax import random

from . import timer
from .data import load


def main():
    interpolator = load.get_bremsstrahlung_interpolator()

    seed = 0
    prng_key = random.PRNGKey(seed)
    num_items = int(1e6)
    iterations = 10
    runs = 10

    iterative_sum = jnp.zeros((num_items,))

    for _ in range(runs):
        prng_key, iterative_sum = _iterations(
            prng_key, iterative_sum, interpolator, iterations
        )


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


# CPU timing on Google Colaboartory:
# _iterations duration: 3868.488 ms
# _iterations duration: 2401.599 ms
# _iterations duration: 2425.017 ms
# _iterations duration: 2532.639 ms
# _iterations duration: 2436.504 ms
# _iterations duration: 2319.901 ms
# _iterations duration: 2390.284 ms
# _iterations duration: 2384.178 ms
# _iterations duration: 2454.509 ms
# _iterations duration: 2433.914 ms

# GPU timing on Google Colaboratory:
# _iterations duration: 3023.384 ms
# _iterations duration: 442.495 ms
# _iterations duration: 451.784 ms
# _iterations duration: 429.913 ms
# _iterations duration: 445.081 ms
# _iterations duration: 445.092 ms
# _iterations duration: 429.942 ms
# _iterations duration: 447.981 ms
# _iterations duration: 452.368 ms
# _iterations duration: 446.351 ms
