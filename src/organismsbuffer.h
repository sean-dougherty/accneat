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
    public:
        OrganismsBuffer(rng_t &rng, size_t n)
            : _n(n) {
            _a.resize(n);
            _b.resize(n);
            _curr = &_a;

            for(size_t i = 0; i < n; i++) {
                _a[i].population_index = i;
                _a[i].genome.rng.seed(rng.integer());
            }
            for(size_t i = 0; i < n; i++) {
                _b[i].population_index = i;
                _b[i].genome.rng.seed(rng.integer());
            }
        }

        size_t size(){
            return _n;
        }

        std::vector<TOrganism> &curr() {
            return *_curr;
        }

        void next_generation(int generation) {
            if(_curr == &_a) {_curr = &_b;} else {_curr = &_a; }
            assert( _curr->size() == _n );

            for(TOrganism &org: curr())
                org.init(generation);
        }

    };

}
