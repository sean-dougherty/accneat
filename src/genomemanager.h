#pragma once

#include "neat.h"
#include "rng.h"
#include <memory>
#include <vector>

namespace NEAT {

    class Genome;

    class GenomeManager {
    public:
        static GenomeManager *create();

        virtual ~GenomeManager() {}

        virtual std::vector<std::unique_ptr<Genome>> create_seed_generation(size_t ngenomes,
                                                                            class rng_t rng,
                                                                            size_t ntraits,
                                                                            size_t ninputs,
                                                                            size_t noutputs,
                                                                            size_t nhidden) = 0;

        virtual void mate(Genome *genome1,
                          Genome *genome2,
                          Genome *offspring,
                          real_t fitness1,
                          real_t fitness2) = 0;
 

        virtual void mutate(Genome *genome) = 0;
    };

}
