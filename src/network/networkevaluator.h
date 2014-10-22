#pragma once

#include "cpunetwork.h"
#include "organism.h"

#define NACTIVATES_PER_INPUT 10

namespace NEAT {

    template<typename Evaluator>
    class NetworkExecutor {
    public:
        
        virtual ~NetworkExecutor() {}

        virtual void configure(const typename Evaluator::Config *config,
                               size_t len) = 0;

        virtual void execute(class Network **nets_,
                             OrganismEvaluation *results,
                             size_t nnets) = 0;
    };

}
