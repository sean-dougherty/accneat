#pragma once

#include "neattypes.h"
#include "networkevaluator.h"

#ifdef ENABLE_CUDA
#include "cudanetworkmanager.h"

namespace NEAT {
}
#else
#include "cpunetworkmanager.h"

namespace NEAT {

    inline std::unique_ptr<class Network> create_default_network() {
        return std::unique_ptr<Network>(new CpuNetwork());
    }

    template<typename Evaluator>
    inline NetworkExecutor<Evaluator> *create_network_executor() {
        return new CpuNetworkExecutor<Evaluator>();
    }

}
#endif
