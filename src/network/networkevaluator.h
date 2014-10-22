#pragma once

#include "cpunetwork.h"
#include "organism.h"

#define NACTIVATES_PER_INPUT 10

namespace NEAT {

    template<typename Evaluator>
    class NetworkExecutor {
    public:
        
        virtual ~NetworkExecutor() {}

        virtual void configure(const typename Evaluator::Config *config,
                               size_t len) = 0;

        virtual void execute(class Network **nets_,
                             OrganismEvaluation *results,
                             size_t nnets) = 0;
    };

    template<typename Evaluator>
    class CpuNetworkExecutor : public NetworkExecutor<Evaluator> {
    public:
        const typename Evaluator::Config *config;

        virtual ~CpuNetworkExecutor() {}

        virtual void configure(const typename Evaluator::Config *config_,
                               size_t len) override {
            config = config_;
        }

        virtual void execute(class Network **nets_,
                             OrganismEvaluation *results,
                             size_t nnets) override {

            CpuNetwork **nets = (CpuNetwork **)nets_;
            size_t nsensors = nets[0]->get_dims().nnodes.sensor;

#pragma omp parallel for
            for(size_t inet = 0; inet < nnets; inet++) {
                CpuNetwork *net = nets[inet];
                Evaluator eval{config};

                for(size_t istep = 0; !eval.complete(istep); istep++) {
                    if(eval.clear_noninput(istep)) {
                        net->clear_noninput();
                    }
                    for(size_t isensor = 0; isensor < nsensors; isensor++) {
                        net->load_sensor(isensor, eval.get_sensor(istep, isensor));
                    }
                    net->activate(NACTIVATES_PER_INPUT);
                    eval.evaluate(istep, net->get_outputs());
                }

                results[inet] = eval.result();
            }
        }
        
    };
}
