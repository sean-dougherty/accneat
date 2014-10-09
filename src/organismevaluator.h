#pragma once

#include "organism.h"

namespace NEAT {

    class OrganismEvaluator {
    public:
        OrganismEvaluator(class Population *pop_);

        Organism *get_fittest();
        bool evaluate(std::function<void (class Organism &org)> eval_org);

    private:
        class Population *pop;
        std::unique_ptr<Organism> fittest;
    };

}
