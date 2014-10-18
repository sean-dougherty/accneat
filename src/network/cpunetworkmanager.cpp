#include "std.h" // Must be included first. Precompiled header with standard library includes.
#include "cpunetworkmanager.h"
#include "cpubatchsensors.h"
#include "cpunetwork.h"
#include "util.h"

using namespace NEAT;
using namespace std;

#define NACTIVATES_PER_INPUT 10

unique_ptr<Network> CpuNetworkManager::make_default() {
    return unique_ptr<Network>(new CpuNetwork());
}

unique_ptr<BatchSensors> CpuNetworkManager::make_batch_sensors(node_size_t nsensors,
                                                               size_t nsteps) {
    return unique_ptr<BatchSensors>(new CpuBatchSensors(nsensors, nsteps));
}

void CpuNetworkManager::activate(class Network **nets,
                                 size_t nnets,
                                 BatchSensors *batch_sensors_,
                                 ProcessOutputFunc process_output) {
    CpuBatchSensors *batch_sensors = dynamic_cast<CpuBatchSensors*>(batch_sensors_);
    size_t nsteps = batch_sensors->get_nsteps();

#pragma omp parallel for
    for(size_t i = 0; i < nnets; i++) {
        CpuNetwork &net = static_cast<CpuNetwork &>(*nets[i]);
        for(size_t istep = 0; istep < nsteps; istep++) {
            batch_sensors->load_sensors(net, istep);
            net.activate(NACTIVATES_PER_INPUT);
            process_output(net, istep);
        }

    }
}
