// compile with: g++ -std=c++11 -pedantic -Wall -O3 -Wextra -o cc-plain cc-plain.cc
// profile with perf stat cc-plain

// includes
#include <iostream>
#include <chrono>
#include <random>
#include <vector>

// defines
#define NUM_PARTICLES 100000
#define ITERATIONS 100
#define RUNS 10

class Particle {
public:
    double x, y, z;
    double u, v, w;
    double E;
};


// main
int main () {

    // title
    std::cout << "--------------------------------------\n";
    std::cout << "C++ with particle array\n";
    std::cout << "--------------------------------------\n";

    // particle array
    std::vector<Particle> particles(NUM_PARTICLES);

    // timing
    auto total_duration = 0;

    // random number generator
    std::mt19937 generator(1);
    std::uniform_real_distribution<double> sample(-1.0, 1.0);

    // update particle arrays a number of times
    for (int run=0; run<RUNS; run++) {

        // poor man's timere
        auto start = std::chrono::high_resolution_clock::now();

        // update particle arrays
        for (int i=0; i<ITERATIONS; i++) {

            for (auto p = particles.begin(); p != particles.end(); ++p) {
                p->x += sample(generator);
                p->y += sample(generator);
                p->z += sample(generator);
                p->u += sample(generator);
                p->v += sample(generator);
                p->w += sample(generator);
                p->E += sample(generator);
            }
        }

        // report duration
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop-start);
        total_duration += duration.count();
        std::cout << "run " << run << ": " << duration.count() << " ms" << std::endl;
    }

    std::cout << "--------------------------------------\n";
    std::cout << "TOTAL = " << total_duration << " ms\n";
    std::cout << "--------------------------------------\n";
    return EXIT_SUCCESS;
}
