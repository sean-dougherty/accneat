#pragma once

#include <cuda.h>

namespace NEAT {

    #define __net_eval_decl __host__ __device__

    class CudaNetwork;

    template<typename Evaluator>
    class CudaNetworkBatch {
    public:
        const typename Evaluator::Config *d_config;

        CudaNetworkBatch(int device_, uint nnets);

        void configure(const typename Evaluator::Config *config,
                       size_t len);
    };

    //---
    //--- CLASS CudaNetworkExecutor
    //---
    template<typename Evaluator>
    class CudaNetworkExecutor : public NetworkExecutor<Evaluator> {
        void __activate(size_t ibatch,
                        CudaNetwork **nets,
                        OrganismEvaluation *results,
                        size_t nnets);

        #define nbatches 3
        class CudaNetworkBatch<Evaluator> *batches[nbatches];
        struct batch_bounds_t {
            size_t i;
            size_t n;
        } batch_bounds [nbatches];
    public:
        CudaNetworkExecutor() {
            abort(); // create batches
        }

        virtual ~CudaNetworkExecutor() {
            for(size_t i = 0; i < nbatches; i++) {
                delete batches[i];
            }
        }

        virtual void configure(const typename Evaluator::Config *config,
                               size_t len) {
            for(size_t i = 0; i < nbatches; i++) {
                batches[i]->configure(config, len);
            }
        }

        virtual void execute(class Network **nets_,
                             OrganismEvaluation *results,
                             size_t nnets) {
            CudaNetwork **nets = (CudaNetwork **)nets_;

#pragma omp parallel for
            for(size_t i = 0; i < nbatches; i++) {
                __activate(i,
                           nets + batch_bounds[i].i,
                           results + batch_bounds[i].i,
                           batch_bounds[i].n);
            }
            
        }
        
    };

    
    template<typename Evaluator>
    inline NetworkExecutor<Evaluator> *NetworkExecutor<Evaluator>::create() {
        return new CudaNetworkExecutor<Evaluator>();
    }
}
