#pragma once

#include "genomemanager.h"
#include "innovation.h"

namespace NEAT {

    class SpaceGenomeManager : public GenomeManager {
    public:
        virtual ~SpaceGenomeManager();

        virtual std::vector<std::unique_ptr<Genome>> create_seed_generation(size_t ngenomes,
                                                                            class rng_t rng,
                                                                            size_t ntraits,
                                                                            size_t ninputs,
                                                                            size_t noutputs,
                                                                            size_t nhidden) override;

        virtual bool are_compatible(Genome &genome1,
                                    Genome &genome2) override;

        virtual void clone(Genome &orig,
                           Genome &clone) override;

        virtual void mate(Genome &genome1,
                          Genome &genome2,
                          Genome &offspring,
                          real_t fitness1,
                          real_t fitness2) override;
 
        virtual void mutate(Genome &genome,
                            MutationOperation op = MUTATE_OP_ANY) override;

        virtual void finalize_generation() override;
    };

}
