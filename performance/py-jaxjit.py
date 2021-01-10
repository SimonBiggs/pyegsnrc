#!/usr/bin/env python

# py-jaxnumpy.py: python/jax.numpy implementation with flat arrays
# requires the JAX library: pip install --upgrade jax jaxlib

import time

import jax

# defines
NUM_PARTICLES = int(1e5)
ITERATIONS = 100
RUNS = 10

# title
print("--------------------------------------")
print("Bare jax.numpy arrays: x = jax.numpy.zeros(NUM_PARTICLES)")
print("--------------------------------------")

# timer decorator
def timer(f):
    def wrap(*args, **kwargs):
        start = time.time()
        ret = f(*args, **kwargs)
        stop = time.time()
        duration = (stop - start) * 1000.0
        print("run {}: {:.3f} ms".format(run, duration))
        return ret

    return wrap


# runIterations
@jax.jit
def runIterations(prng_key, x, y, z, u, v, w, E):
    for j in range(ITERATIONS):

        random_values = jax.random.normal(prng_key, shape=(7, NUM_PARTICLES))
        (prng_key,) = jax.random.split(prng_key, 1)

        x += random_values[0, :]
        y += random_values[1, :]
        z += random_values[2, :]
        u += random_values[3, :]
        v += random_values[4, :]
        w += random_values[5, :]
        E += random_values[6, :]

    return prng_key, x, y, z, u, v, w, E


# timed version of runIterations
runIterations = timer(runIterations)

# main function
def main(run):

    seed = 0
    prng_key = jax.random.PRNGKey(seed)

    x = jax.numpy.zeros(NUM_PARTICLES)
    y = jax.numpy.zeros(NUM_PARTICLES)
    z = jax.numpy.zeros(NUM_PARTICLES)
    u = jax.numpy.zeros(NUM_PARTICLES)
    v = jax.numpy.zeros(NUM_PARTICLES)
    w = jax.numpy.zeros(NUM_PARTICLES)
    E = jax.numpy.zeros(NUM_PARTICLES)

    (prng_key, x, y, z, u, v, w, E) = runIterations(prng_key, x, y, z, u, v, w, E)


# call main function
for run in range(RUNS):
    main(run)

print("--------------------------------------")
