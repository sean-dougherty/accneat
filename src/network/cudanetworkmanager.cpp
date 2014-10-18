#ifdef ENABLE_CUDA

#include "std.h" // Must be included first. Precompiled header with standard library includes.
#include "cudanetworkmanager.h"
#include "cudanetwork.h"
#include "neat.h"
#include "util.h"
#include <assert.h>
#include <omp.h>

using namespace NEAT;
using namespace std;

#define NACTIVATES_PER_INPUT 10

CudaNetworkManager::CudaNetworkManager() {
    size_t batch_size = env->pop_size / nbatches;

    for(size_t i = 0; i < nbatches; i++) {
        batch_bounds[i].i = i * batch_size;
        if( i == (nbatches - 1) ) {
            batch_size += env->pop_size % nbatches;
        }
        batch_bounds[i].n = batch_size;

        batches[i] = new CudaNetworkBatch(i, batch_size);
    }
}

unique_ptr<Network> CudaNetworkManager::make_default() {
    return unique_ptr<Network>(new CudaNetwork());
}

void CudaNetworkManager::activate(Network **nets_, size_t nnets,
                                  BatchSensors *batch_sensors_,
                                  ProcessOutputFunc process_output) {
    CudaNetwork **nets = (CudaNetwork **)nets_;
    CudaBatchSensors *batch_sensors = (CudaBatchSensors *)batch_sensors_;

#pragma omp parallel for
    for(size_t i = 0; i < nbatches; i++) {
        __activate(i,
                   nets + batch_bounds[i].i, batch_bounds[i].n,
                   batch_sensors, process_output);
    }
}

void CudaNetworkManager::__activate(size_t ibatch,
                                    CudaNetwork **nets, size_t nnets,
                                    CudaBatchSensors *batch_sensors,
                                    ProcessOutputFunc process_output) {
    CudaNetworkBatch *batch = batches[ibatch];

    batch->configure(batch_sensors, nets, nnets);
    batch->activate(NACTIVATES_PER_INPUT);

    size_t nsteps = batch_sensors->get_dims().nsteps;
    for(size_t istep = 0; istep < nsteps; istep++) {
        batch->set_output_step(istep);
        for(size_t inet = 0; inet < nnets; inet++) {
            process_output(*nets[inet], istep);
        }
    }
}

unique_ptr<BatchSensors> CudaNetworkManager::make_batch_sensors(node_size_t nsensors,
                                                                size_t nsteps) {
    return unique_ptr<BatchSensors>(new CudaBatchSensors({nsensors, nsteps}));
}

#endif // #if ENABLE_CUDA
