#!/usr/bin/env python

# py-jaxornot.py: python/jax.numpy implementation with flat arrays
# requires the JAX library: pip install --upgrade jax jaxlib

# use jax or not?
USE_JAX = False

import time

if USE_JAX:
    import jax
    import jax.numpy as numpy
    import jax.random as random
else:
    import numpy
    import numpy.random as random

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
def runIterations(prng_key, x, y, z, u, v, w, E):
    for j in range(ITERATIONS):

        if USE_JAX:
            random_values = random.normal(prng_key, shape=(7, NUM_PARTICLES))
            (prng_key,) = random.split(prng_key, 1)
        else:
            random_values = numpy.random.rand(7, NUM_PARTICLES)

        x += random_values[0, :]
        y += random_values[1, :]
        z += random_values[2, :]
        u += random_values[3, :]
        v += random_values[4, :]
        w += random_values[5, :]
        E += random_values[6, :]

    return prng_key, x, y, z, u, v, w, E


# jit compilation if using jax
if USE_JAX:
    runIterations = jax.jit(runIterations)

# timed version of runIterations
runIterations = timer(runIterations)

# main function
def main(run):

    seed = 0
    if USE_JAX:
        prng_key = random.PRNGKey(seed)
    else:
        prng_key = random.seed(seed)

    x = numpy.zeros(NUM_PARTICLES)
    y = numpy.zeros(NUM_PARTICLES)
    z = numpy.zeros(NUM_PARTICLES)
    u = numpy.zeros(NUM_PARTICLES)
    v = numpy.zeros(NUM_PARTICLES)
    w = numpy.zeros(NUM_PARTICLES)
    E = numpy.zeros(NUM_PARTICLES)

    (prng_key, x, y, z, u, v, w, E) = runIterations(prng_key, x, y, z, u, v, w, E)


# call main function
for run in range(RUNS):
    main(run)

print("--------------------------------------")
