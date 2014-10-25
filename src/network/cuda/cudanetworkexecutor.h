#pragma once

#include "networkexecutor.h"
#include "cudanetworkbatch.h"

namespace NEAT {

    //---
    //--- CLASS CudaNetworkExecutor
    //---
    template<typename Evaluator>
    class CudaNetworkExecutor : public NetworkExecutor<Evaluator> {
        std::vector<class CudaNetworkBatch<Evaluator> *> batches;
    public:
        CudaNetworkExecutor() {
            int ndevices;
            xcuda( cudaGetDeviceCount(&ndevices) );
            errif(ndevices == 0, "No Cuda devices found!");

            batches.resize(ndevices);
            for(int i = 0; i < ndevices; i++) {
                batches[i] = new CudaNetworkBatch<Evaluator>(i);
            }
        }

        virtual ~CudaNetworkExecutor() {
            for(size_t i = 0; i < batches.size(); i++) {
                delete batches[i];
            }
        }

        virtual void configure(const typename Evaluator::Config *config,
                               size_t len) {
            for(size_t i = 0; i < batches.size(); i++) {
                batches[i]->configure(config, len);
            }
        }

        virtual void execute(class Network **nets_,
                             OrganismEvaluation *results,
                             size_t nnets) {
            CudaNetwork **nets = (CudaNetwork **)nets_;
            size_t nbatches = batches.size();
            uint batch_size = nnets / nbatches;

#pragma omp parallel for
            for(size_t ibatch = 0; ibatch < nbatches; ibatch++) {
                size_t inet = ibatch * batch_size;
                size_t n = batch_size;
                if(ibatch == nbatches - 1)
                    n += nnets % batch_size;

                batches[ibatch]->activate(nets + inet,
                                          results + inet,
                                          n,
                                          NACTIVATES_PER_INPUT);
            }
            
        }
        
    };

    template<typename Evaluator>
    inline NetworkExecutor<Evaluator> *NetworkExecutor<Evaluator>::create() {
        return new CudaNetworkExecutor<Evaluator>();
    }

} // namespace NEAT
