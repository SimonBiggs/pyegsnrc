#!/usr/bin/env python

# py-plain.py: plain python/numpy implementation with flat arrays

import numpy
import time

# defines
NUM_PARTICLES = int(1e5)
ITERATIONS = 100
RUNS = 10

# title
print("--------------------------------------")
print("Python with plain numpy arrays: x, y, z, u, v, w, E")
print("--------------------------------------")

# timer decorator
def timer(f):
    def wrap(*args, **kwargs):
        start = time.time()
        ret = f(*args, **kwargs)
        stop = time.time()
        duration = (stop-start)*1000.0
        print('run {}: {:.3f} ms'.format(run, duration))
        return ret
    return wrap

# runIterations
def runIterations(x, y, z, u, v, w, E):
    for j in range(ITERATIONS):
        random_normal = numpy.random.rand(NUM_PARTICLES)
        x += random_normal
        random_normal = numpy.random.rand(NUM_PARTICLES)
        y += random_normal
        random_normal = numpy.random.rand(NUM_PARTICLES)
        z += random_normal
        random_normal = numpy.random.rand(NUM_PARTICLES)
        u += random_normal
        random_normal = numpy.random.rand(NUM_PARTICLES)
        v += random_normal
        random_normal = numpy.random.rand(NUM_PARTICLES)
        w += random_normal
        random_normal = numpy.random.rand(NUM_PARTICLES)
        E += random_normal

# timed version of runIterations
runIterations = timer(runIterations)

# main function
def main(run):

    numpy.random.seed()

    x = numpy.zeros(NUM_PARTICLES)
    y = numpy.zeros(NUM_PARTICLES)
    z = numpy.zeros(NUM_PARTICLES)
    u = numpy.zeros(NUM_PARTICLES)
    v = numpy.zeros(NUM_PARTICLES)
    w = numpy.zeros(NUM_PARTICLES)
    E = numpy.zeros(NUM_PARTICLES)

    runIterations(x, y, z, u, v, w, E)

# call main function
for run in range(RUNS):
    main(run)


print("--------------------------------------")