#include "std.h"
#include "organismevaluator.h"
#include "population.h"
#include "timer.h"
#include <omp.h>

#define NACTIVATES_PER_INPUT 10

using namespace NEAT;
using namespace std;

OrganismEvaluator::OrganismEvaluator(Population *pop_)
    : pop(pop_) {
}

Organism *OrganismEvaluator::get_fittest() {
    return fittest.get();
}

bool OrganismEvaluator::evaluate(std::function<bool (Organism &org, size_t istep)> prepare_step,
                                 std::function<void (Organism &org, size_t istep)> eval_step,
                                 std::function<OrganismEvaluation (Organism &org)> eval) {
    static Timer timer("evaluate");
    timer.start();

    size_t norgs = pop->size();

#if true
#pragma omp parallel for
    for(size_t i = 0; i < norgs; i++) {
        Organism *org = pop->get(i);

        for(size_t istep = 0; prepare_step(*org, istep); istep++) {
            for(size_t j = 0; j < NACTIVATES_PER_INPUT; j++) {
                org->net->activate();
            }
            eval_step(*org, istep);
        }

        org->eval = eval(*org);
    }
#else
    bool remaining = true;
    for(size_t istep = 0; remaining; istep++) {
        remaining = false;
#pragma omp parallel for reduction(||:remaining)
        for(size_t iorg = 0; iorg < norgs; iorg++) {
            Organism *org = pop->get(iorg);
            if(prepare_step(*org, istep)) {
                remaining = true;
                for(size_t j = 0; j < NACTIVATES_PER_INPUT; j++) {
                    org->net.activate();
                }
                eval_step(*org, istep);
            } else {
                org->eval = eval(*org);
            }
        }
    }
#endif

    Organism *best = pop->get(0);
    for(size_t i = 1; i < norgs; i++) {
        Organism *org = pop->get(i);
        if(org->eval.fitness > best->eval.fitness) {
            best = org;
        }
    }

    timer.stop();

    if(!fittest || (best->eval.fitness > fittest->eval.fitness)) {
        fittest = pop->make_copy(best->population_index);
        return true;
    } else {
        return false;
    }
}
