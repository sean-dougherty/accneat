#pragma once

#include "organism.h"
#include "rng.h"
#include <vector>

namespace NEAT {

    class OrganismsBuffer {
        size_t _n;
        std::vector<Organism> _a;
        std::vector<Organism> _b;
        std::vector<Organism> *_curr;
    public:
        OrganismsBuffer(rng_t &rng, size_t n);

        size_t size();
        std::vector<Organism> &curr();
        void next_generation(int generation);
    };

}
