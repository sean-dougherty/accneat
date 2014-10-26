#pragma once

#include "cpunetwork.h"

namespace NEAT {

    //Don't need any special qualifiers for CPU
    #define __net_eval_decl

    //---
    //--- CLASS CpuNetworkExecutor
    //---
    template<typename Evaluator>
    class CpuNetworkExecutor : public NetworkExecutor<Evaluator> {
    public:
        const typename Evaluator::Config *config;

        CpuNetworkExecutor() {
            config = NULL;
        }

        virtual ~CpuNetworkExecutor() {
            delete config;
        }

        virtual void configure(const typename Evaluator::Config *config_,
                               size_t len) {
            void *buf = malloc(len);
            memcpy(buf, config_, len);
            config = (const typename Evaluator::Config *)buf;
        }

        virtual void execute(class Network **nets_,
                             OrganismEvaluation *results,
                             size_t nnets) {

            CpuNetwork **nets = (CpuNetwork **)nets_;
            size_t nsensors = nets[0]->get_dims().nnodes.sensor;

#pragma omp parallel for
            for(size_t inet = 0; inet < nnets; inet++) {
                CpuNetwork *net = nets[inet];
                Evaluator eval(config);

                while(eval.next_step()) {
                    if(eval.clear_noninput()) {
                        net->clear_noninput();
                    }
                    for(size_t isensor = 0; isensor < nsensors; isensor++) {
                        net->load_sensor(isensor, eval.get_sensor(isensor));
                    }
                    net->activate(NACTIVATES_PER_INPUT);
                    eval.evaluate(net->get_outputs());
                }

                results[inet] = eval.result();
            }
        }
        
    };

    //---
    //--- FUNC NetworkExecutor<Evaluator>::create()
    //---
    template<typename Evaluator>
    inline NetworkExecutor<Evaluator> *NetworkExecutor<Evaluator>::create() {
        return new CpuNetworkExecutor<Evaluator>();
    }
}
