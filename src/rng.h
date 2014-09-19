#pragma once

#include "neat.h"
#include <climits>
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

        int integer(int low = INT_MIN, int hi = INT_MAX) {
            std::uniform_int_distribution<int> dist(low, hi);
            return dist(engine);
        }

        // value in [0,1] from uniform distribution
        real_t prob() {
            std::uniform_real_distribution<real_t> dist(0, 1);
            return dist(engine);
        }

        bool under(real_t prob) {
            return prob <= 0.0 ? false : this->prob() < prob;
        }

        class prob_switch_t {
            real_t x;
            friend class rng_t;
            prob_switch_t(real_t x_) : x(x_) {}
        public:
            bool prob_case(real_t prob) {
                if(x < prob) {
                    return true;
                } else {
                    x -= prob;
                    return false;
                }
            }
        };

        prob_switch_t prob_switch() {
            return prob_switch_t(prob());
        }

        // -1 or 1
        int posneg() {
            std::uniform_int_distribution<int> dist(0, 1);
            return dist(engine) * 2 - 1;
        }

        // value from z distribution
        real_t gauss() {
            std::normal_distribution<real_t> dist;
            return dist(engine);
        }

        static void test();
    };
}
