#pragma once

#include "networkmanager.h"
#include "organism.h"

namespace NEAT {

    class OrganismEvaluator {
    public:
        OrganismEvaluator(class Population *pop_);

        Organism *get_fittest();

        bool evaluate(NetworkManager::LoadSensorsFunc load_sensors,
                      NetworkManager::ProcessOutputFunc process_output,
                      std::function<OrganismEvaluation (Organism &org)> eval);

    private:
        class Population *pop;
        std::unique_ptr<Organism> fittest;
    };

}
