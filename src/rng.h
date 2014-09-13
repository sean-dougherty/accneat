#pragma once

#include <random>
#include <vector>

namespace NEAT {
    class rng_t {
        std::default_random_engine engine;

    public:
        void seed(int sval) {
            engine.seed(sval);
        }

        template<typename T>
            size_t index(std::vector<T> &v, size_t begin = 0) {
            std::uniform_int_distribution<int> dist(begin, v.size() - 1);
            return dist(engine);
        }

        template<typename T>
            T& element(std::vector<T> &v, size_t begin = 0) {
            return v[index(v, begin)];
        }

        // value in [0,1] from uniform distribution
        double prob() {
            std::uniform_real_distribution<double> dist(0, 1);
            return dist(engine);
        }

        // -1 or 1
        int posneg() {
            std::uniform_int_distribution<int> dist(0, 1);
            return dist(engine) * 2 - 1;
        }

        // value from z distribution
        double gauss() {
            std::normal_distribution<double> dist;
            return dist(engine);
        }

        static void test();
    };
}
