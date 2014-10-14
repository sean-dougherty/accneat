#include "std.h" // Must be included first. Precompiled header with standard library includes.
#include "cpunetworkmanager.h"
#include "cpunetwork.h"

using namespace NEAT;
using namespace std;

#define NACTIVATES_PER_INPUT 10

#define SERIAL_EXECUTION false

unique_ptr<Network> CpuNetworkManager::make_default() {
    return unique_ptr<Network>(new CpuNetwork());
}

void CpuNetworkManager::activate(Network **nets, size_t nnets,
                                 LoadSensorsFunc load_sensors,
                                 ProcessOutputFunc process_output) {
#if SERIAL_EXECUTION
//---
//--- Serial
//---

//Does a better job keeping neural nets in the CPU cache, which should help
//performance. On my underpowered laptop, the difference is noticeable. On
//a high-end workstation, it doesn't offer a perceivable boost.
#pragma omp parallel for
    for(size_t i = 0; i < nnets; i++) {
        CpuNetwork &net = static_cast<CpuNetwork &>(*nets[i]);
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
        for(size_t inet = 0; inet < nnets; inet++) {
            CpuNetwork &net = static_cast<CpuNetwork &>(*nets[inet]);
            if(load_sensors(net, istep)) {
                remaining = true;
                net.activate(NACTIVATES_PER_INPUT);
                process_output(net, istep);
            }
        }
    }
#endif

}
