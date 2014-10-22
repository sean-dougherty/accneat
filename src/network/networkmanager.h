#pragma once

#include "neattypes.h"
#include "networkevaluator.h"

namespace NEAT {

    class NetworkManager {
    public:
        //---
        //--- Interface
        //---
        virtual ~NetworkManager() {}

        virtual std::unique_ptr<class Network> make_default() = 0;
    };

}

#ifdef ENABLE_CUDA
#include "cudanetworkmanager.h"

namespace NEAT {

    inline NetworkManager *create_network_manager() {
        return new CudaNetworkManager();
    }

}
#else
#include "cpunetworkmanager.h"

namespace NEAT {

    inline NetworkManager *create_network_manager() {
        return new CpuNetworkManager();
    }

    template<typename Evaluator>
    inline NetworkExecutor<Evaluator> *create_executor() {
        return new CpuNetworkExecutor<Evaluator>();
    }

}
#endif
