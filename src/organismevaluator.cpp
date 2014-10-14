#include "std.h"
#include "networkmanager.h"
#include "organismevaluator.h"
#include "population.h"
#include "timer.h"

using namespace NEAT;
using namespace std;

OrganismEvaluator::OrganismEvaluator(Population *pop_)
    : pop(pop_) {
}

Organism *OrganismEvaluator::get_fittest() {
    return fittest.get();
}

bool OrganismEvaluator::evaluate(NetworkManager::LoadSensorsFunc load_sensors,
                                 NetworkManager::ProcessOutputFunc process_output,
                                 std::function<OrganismEvaluation (Organism &org)> eval) {
    static Timer timer("evaluate");
    timer.start();

    size_t norgs = pop->size();
    Network *nets[norgs];
    for(size_t i = 0; i < norgs; i++) {
        nets[i] = pop->get(i)->net.get();
    }

    env->network_manager->activate(nets, norgs, load_sensors, process_output);

    Organism *best = nullptr;
    for(size_t i = 0; i < norgs; i++) {
        Organism *org = pop->get(i);
        org->eval = eval(*org);
        if( !best || (org->eval.fitness > best->eval.fitness) ) {
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
