#pragma once

#include "genomemanager.h"
#include "innovation.h"

namespace NEAT {

    class InnovGenomeManager : public GenomeManager {
        friend class GenomeManager;
        InnovGenomeManager();
    public:
        virtual ~InnovGenomeManager();

        virtual std::unique_ptr<Genome> make_default() override;

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

    private:
        CreateInnovationFunc create_innov_func(Genome &g);
        bool is_mate_allowed();
        bool is_add_allowed();
        bool is_delete_allowed();

        PopulationInnovations innovations;

        int generation;
        enum SearchPhase {
            UNDEFINED,
            COMPLEXIFY,
            PRUNE
        } search_phase;
        int search_phase_start;
    };

}
