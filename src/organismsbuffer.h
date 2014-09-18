#pragma once

#include "organism.h"
#include "rng.h"
#include <vector>

namespace NEAT {

    template<typename TOrganism = Organism>
    class OrganismsBuffer {
        size_t _n;
        std::vector<TOrganism> _a;
        std::vector<TOrganism> _b;
        std::vector<TOrganism> *_curr;
        std::vector<TOrganism> *_prev;
    public:
    OrganismsBuffer(rng_t &rng, size_t n, size_t population_index = 0)
            : _n(n) {
            _a.resize(n);
            _b.resize(n);
            _curr = &_a;
            _prev = &_b;

            for(size_t i = 0; i < n; i++) {
                _a[i].population_index = i + population_index;
                _a[i].genome.rng.seed(rng.integer());
            }
            for(size_t i = 0; i < n; i++) {
                _b[i].population_index = i + population_index;
                _b[i].genome.rng.seed(rng.integer());
            }
        }

        size_t size(){
            return _n;
        }

        std::vector<TOrganism> &curr() {
            return *_curr;
        }

        std::vector<TOrganism> &prev() {
            return *_prev;
        }

        void next_generation(int generation) {
            if(_curr == &_a) {
                _curr = &_b;
                _prev = &_a;
            } else {
                _curr = &_a;
                _prev = &_b;
            }

            assert( _curr->size() == _n );

            for(TOrganism &org: curr())
                org.init(generation);
        }

    };

}
