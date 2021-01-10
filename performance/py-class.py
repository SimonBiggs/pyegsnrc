#!/usr/bin/env python

# py-class.py: try to process particles as an array of Particle class objects

import time

import numpy

# defines
NUM_PARTICLES = int(1e5)
ITERATIONS = 10
RUNS = 10

# particle class
class Particle:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.z = 0
        self.u = 0
        self.v = 0
        self.w = 0
        self.E = 0

    def update(self, random_values):
        self.x += random_values[0]
        self.y += random_values[1]
        self.z += random_values[2]
        self.u += random_values[3]
        self.v += random_values[4]
        self.w += random_values[5]
        self.E += random_values[6]


# title
print("--------------------------------------")
print(
    "Python with list of particle class: particles = [Particle() for _ in range(NUM_PARTICLES)]"
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
        for p in particles:
            p.update(numpy.random.rand(7))


# timed version of runIterations
runIterations = timer(runIterations)

# main function
def main(run):

    numpy.random.seed()
    particles = [Particle() for _ in range(NUM_PARTICLES)]
    runIterations(particles)


# call main function
for run in range(RUNS):
    main(run)


print("--------------------------------------")
