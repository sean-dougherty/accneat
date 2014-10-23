#pragma once

namespace NEAT {

#define NACTIVATES_PER_INPUT 10

    //---
    //--- CLASS NetworkExecutor<>
    //---
    template<typename Evaluator>
    class NetworkExecutor {
    public:
        
        virtual ~NetworkExecutor() {}

        virtual void configure(const typename Evaluator::Config *config,
                               size_t len) = 0;

        virtual void execute(class Network **nets_,
                             class OrganismEvaluation *results,
                             size_t nnets) = 0;
    };

}

#ifdef ENABLE_CUDA
#include "cudanetworkexecutor.h"
#else
#include "cpunetworkexecutor.h"
#endif
