#pragma once

#include "organism.h"

namespace NEAT {

    class OrganismEvaluator {
    public:
        OrganismEvaluator(class Population *pop_);

        Organism *get_fittest();

        bool evaluate(std::function<bool (Organism &org, size_t istep)> prepare_step,
                      std::function<void (Organism &org, size_t istep)> eval_step,
                      std::function<OrganismEvaluation (Organism &org)> eval);

    private:
        class Population *pop;
        std::unique_ptr<Organism> fittest;
    };

}
