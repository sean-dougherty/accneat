#ifdef ENABLE_CUDA

#include "std.h" // Must be included first. Precompiled header with standard library includes.
#include "cudanetworkmanager.h"
#include "cudanetwork.h"
#include "neat.h"
#include "util.h"

using namespace NEAT;
using namespace std;

#define NACTIVATES_PER_INPUT 10

CudaNetworkManager::CudaNetworkManager() {
    batch = new CudaNetworkBatch(env->pop_size);
}

unique_ptr<Network> CudaNetworkManager::make_default() {
    return unique_ptr<Network>(new CudaNetwork());
}

void CudaNetworkManager::activate(Network **nets_, size_t nnets,
                                  LoadSensorsFunc load_sensors,
                                  ProcessOutputFunc process_output) {
    CudaNetwork **nets = (CudaNetwork **)nets_;

    batch->configure(nets, nnets);

    bool remaining = true;
    for(size_t istep = 0; remaining; istep++) {
        remaining = false;
        for(size_t inet = 0; inet < nnets; inet++) {
            if(nets[inet]->is_enabled() && !load_sensors(*nets[inet], istep)) {
                nets[inet]->disable();
            } else {
                remaining = true;
            }
        }

        if(remaining) {
            batch->activate(NACTIVATES_PER_INPUT);

            for(size_t inet = 0; inet < nnets; inet++) {
                if(nets[inet]->is_enabled()) {
                    process_output(*nets[inet], istep);
                }
            }
        }
    }
    
}

#endif
