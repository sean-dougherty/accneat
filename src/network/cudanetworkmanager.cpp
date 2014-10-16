#ifdef ENABLE_CUDA

#include "std.h" // Must be included first. Precompiled header with standard library includes.
#include "cudanetworkmanager.h"
#include "cudanetwork.h"
#include "neat.h"
#include "util.h"

//#define VERIFY_VIA_CPU

#ifdef VERIFY_VIA_CPU
#include "cpunetwork.h"
#include "organism.h"
#include "population.h"
#include <assert.h>
#endif

using namespace NEAT;
using namespace std;

#define NACTIVATES_PER_INPUT 10

ostream &operator<<(ostream &out, const vector<real_t> &x) {
    out << "{";
    for(size_t i = 0; i < x.size(); i++) {
        if(i != 0) out << ", ";
        out << x[i];
    }
    out << "}";
    return out;
}

CudaNetworkManager::CudaNetworkManager() {
    //test_sum_partition();
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

#ifdef VERIFY_VIA_CPU
/*************************************************************/
/**/static size_t ii = 0;
/**/CpuNetwork cpunets[nnets];
/**/for(size_t i = 0; i < nnets; i++) {
/**/    debug_population->get(i)->genome->init_phenotype(cpunets[i]);
/**/    cpunets[i].population_index = i;
/**/}
/*************************************************************/
#endif

    bool remaining = true;
    for(size_t istep = 0; remaining; istep++) {
        remaining = false;

#pragma omp parallel for reduction(||:remaining)
        for(size_t inet = 0; inet < nnets; inet++) {
            if(nets[inet]->is_enabled() && !load_sensors(*nets[inet], istep)) {
                nets[inet]->disable();
            } else {
#ifdef VERIFY_VIA_CPU
/*************************************************************/
/**/            load_sensors(cpunets[inet], istep);
/*************************************************************/
#endif
                remaining = true;
            }
        }

        if(remaining) {
#ifndef VERIFY_VIA_CPU
            batch->activate(NACTIVATES_PER_INPUT);
#else
/*************************************************************/
/**/        vector<real_t> cpu_act;
/**/        vector<real_t> cuda_act;
/**/
/**/        for(size_t icycle = 0; icycle < NACTIVATES_PER_INPUT / 10; icycle++) {
/**/            ii++;
/**/
/**/            batch->activate(10);
/**/            for(size_t inet = 0; inet < nnets; inet++) {
/**/                nets[inet]->set_clear_noninput(false);
/**/            }
/**/            //cout << "[" << ii << "] gpu=" << batch->get_activations(nets[0], cuda_act) << endl;
/**/
/**/            for(size_t inet = 0; inet < nnets; inet++) {
/**/                cpunets[inet].activate(10);
/**/                //cout << "[" << ii << "] cpu=" << cpunets[inet].get_activations(cpu_act) << endl;
/**/
/**/                cpunets[inet].get_activations(cpu_act);
/**/                batch->get_activations(nets[inet], cuda_act);
/**/
/**/                assert(cpu_act.size() == cuda_act.size());
/**/                for(size_t iact = 0; iact < cpu_act.size(); iact++) {
/**/                    if( fabs(cpu_act[iact] - cuda_act[iact]) > 0.05 ) {
/**/                        cout << "[" << ii << "] mismatch at index " << iact << endl;
/**/                        cout << "cpu =" << cpu_act << endl;
/**/                        cout << "cuda=" << cuda_act << endl;
/**/                        getchar();
/**/                    }
/**/                }
/**/
/**/                cpunets[inet].set_activations(cuda_act);
/**/            }
/**/        }
/*************************************************************/
#endif

#pragma omp parallel for
            for(size_t inet = 0; inet < nnets; inet++) {
                if(nets[inet]->is_enabled()) {
                    process_output(*nets[inet], istep);
                }
            }
        }
    }
    
}

#endif
