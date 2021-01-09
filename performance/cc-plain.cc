// compile with: g++ -std=c++11 -pedantic -Wall -O3 -Wextra -o cc-plain cc-plain.cc
// profile with perf stat cc-plain

// includes
#include <iostream>
#include <chrono>
#include <random>

// defines
#define NUM_PARTICLES 100000
#define ITERATIONS 100
#define RUNS 10

// main
int main () {

    // positions
    double *x = new double[NUM_PARTICLES];
    double *y = new double[NUM_PARTICLES];
    double *z = new double[NUM_PARTICLES];

    // directions
    double *u = new double[NUM_PARTICLES];
    double *v = new double[NUM_PARTICLES];
    double *w = new double[NUM_PARTICLES];

    // energies
    double *E = new double[NUM_PARTICLES];

    // timing
    auto total_duration = 0;

    // random number generator
    std::mt19937 generator(1);
    std::uniform_real_distribution<double> sample(-1.0, 1.0);

    // update particle arrays a number of times
    for (int run=0; run<RUNS; run++) {

        // poor man's timer
        auto start = std::chrono::high_resolution_clock::now();

        // update particle arrays
        for (int i=0; i<ITERATIONS; i++) {
            for (int n=0; n<NUM_PARTICLES; n++) {
                x[n] += sample(generator);
                y[n] += sample(generator);
                z[n] += sample(generator);
                u[n] += sample(generator);
                v[n] += sample(generator);
                w[n] += sample(generator);
                E[n] += sample(generator);
            }
        }

        // report duration
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop-start);
        total_duration += duration.count();
        std::cout << "run " << run << ": duration = " << duration.count() << " ms" << std::endl;
    }

    std::cout << "--------------------------------------\n";
    std::cout << "TOTAL = " << total_duration << " ms\n";
    std::cout << "--------------------------------------\n";
    return EXIT_SUCCESS;
}