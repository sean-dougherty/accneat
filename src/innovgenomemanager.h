#pragma once

#include "genomemanager.h"
#include "innovation.h"

namespace NEAT {

    class InnovGenomeManager : public GenomeManager {
    public:
        virtual ~InnovGenomeManager();

        virtual std::vector<std::unique_ptr<Genome>> create_seed_generation(size_t ngenomes,
                                                                            class rng_t rng,
                                                                            size_t ntraits,
                                                                            size_t ninputs,
                                                                            size_t noutputs,
                                                                            size_t nhidden) override;

        virtual void mate(Genome *genome1,
                          Genome *genome2,
                          Genome *offspring,
                          real_t fitness1,
                          real_t fitness2) override;
 

        virtual void mutate(Genome *genome) override;

        //private: todo: uncomment
        PopulationInnovations innovations;        
    };

}
