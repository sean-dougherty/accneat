#include "std.h" // Must be included first. Precompiled header with standard library includes.
#include "cpunetworkmanager.h"
#include "cpunetwork.h"

using namespace NEAT;
using namespace std;

#define NACTIVATES_PER_INPUT 10

unique_ptr<Network> CpuNetworkManager::make_default() {
    return unique_ptr<Network>(new CpuNetwork());
}

void CpuNetworkManager::activate(Network **nets, size_t nnets,
                                 LoadSensorsFunc load_sensors,
                                 ProcessOutputFunc process_output) {
#if true
//---
//--- Serial
//---
#pragma omp parallel for
    for(size_t i = 0; i < nnets; i++) {
        Network &net = *nets[i];
        for(size_t istep = 0; load_sensors(net, istep); istep++) {
            net.activate(NACTIVATES_PER_INPUT);
            process_output(net, istep);
        }

    }
#else
//---
//--- Parallel
//---
    bool remaining = true;
    for(size_t istep = 0; remaining; istep++) {
        remaining = false;
#pragma omp parallel for reduction(||:remaining)
        for(size_t iorg = 0; iorg < norgs; iorg++) {
            Organism *org = pop->get(iorg);
            if(load_sensors(*org->net, istep)) {
                remaining = true;
                org->net->activate(NACTIVATES_PER_INPUT);
                process_output(*org->net, istep);
            } else {
                org->eval = eval(*org);
            }
        }
    }
#endif

}
