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

    size_t nthreads = omp_get_max_threads();
    Organism *fittest_thread[nthreads];

    for(size_t i = 0; i < nthreads; i++)
        fittest_thread[i] = nullptr;

    size_t norgs = pop->size();
#pragma omp parallel for
    for(size_t i = 0; i < norgs; i++) {
        Organism *org = pop->get(i);

        for(size_t istep = 0; prepare_step(*org, istep); istep++) {
            for(size_t j = 0; j < NACTIVATES_PER_INPUT; j++) {
                org->net.activate();
            }
            eval_step(*org, istep);
        }

        org->eval = eval(*org);

        size_t tnum = omp_get_thread_num();
        if( (fittest_thread[tnum] == nullptr)
            || (org->eval.fitness > fittest_thread[tnum]->eval.fitness) ) {

            fittest_thread[tnum] = org;
        }
    }

    Organism *best = nullptr;
    for(size_t i = 0; i < nthreads; i++) {
        if( !best
            || (fittest_thread[i]
                && (fittest_thread[i]->eval.fitness > best->eval.fitness)) ) {

            best = fittest_thread[i];
        }
    }

    timer.stop();

    if( best &&
       (!fittest || (best->eval.fitness > fittest->eval.fitness)) ) {

        fittest = pop->make_copy(best->population_index);
        return true;
    } else {
        return false;
    }
}
