#include "std.h"
#include "organismevaluator.h"
#include "population.h"
#include "timer.h"
#include <omp.h>

using namespace NEAT;
using namespace std;

OrganismEvaluator::OrganismEvaluator(Population *pop_)
    : pop(pop_) {
}

Organism *OrganismEvaluator::get_fittest() {
    return fittest.get();
}

bool OrganismEvaluator::evaluate(std::function<void (Organism &org)> eval_org) {
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
        eval_org( *org );

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
