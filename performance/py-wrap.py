#!/usr/bin/env python

# py-wrap.py: python/numpy implementation with flat arrays, but wrapped in an
# object api.

import time

import numpy

# defines
NUM_PARTICLES = int(1e5)
ITERATIONS = 100
RUNS = 1

# particle class
class ParticleArray:
    def __init__(self):
        self.x = numpy.zeros(NUM_PARTICLES)
        self.y = numpy.zeros(NUM_PARTICLES)
        self.z = numpy.zeros(NUM_PARTICLES)
        self.u = numpy.zeros(NUM_PARTICLES)
        self.v = numpy.zeros(NUM_PARTICLES)
        self.w = numpy.zeros(NUM_PARTICLES)
        self.E = numpy.zeros(NUM_PARTICLES)

    def update(self, random_values):
        self.x += random_values[0, :]
        self.y += random_values[1, :]
        self.z += random_values[2, :]
        self.u += random_values[3, :]
        self.v += random_values[4, :]
        self.w += random_values[5, :]
        self.E += random_values[6, :]


# title
print("--------------------------------------")
print(
    "Python class wrapped plain numpy arrays: class Particle: self.x = numpy.zeros(NUM_PARTICLES) etc."
)
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
def runIterations(particles):
    for j in range(ITERATIONS):
        random_values = numpy.random.rand(7, NUM_PARTICLES)
        particles.update(random_values)


# timed version of runIterations
runIterations = timer(runIterations)

# main function
def main(run):
    numpy.random.seed()
    particles = ParticleArray()
    runIterations(particles)


# call main function
for run in range(RUNS):
    main(run)


print("--------------------------------------")
