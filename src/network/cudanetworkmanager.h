#pragma once

#include "networkmanager.h"

namespace NEAT {

    class CudaNetworkManager : public NetworkManager {
    public:
        CudaNetworkManager();
        virtual ~CudaNetworkManager() {}

        virtual std::unique_ptr<class Network> make_default() override;

        virtual void activate(class Network **nets, size_t nnets,
                              LoadSensorsFunc load_sensors,
                              ProcessOutputFunc process_output) override;


    private:
        void __activate(size_t ibatch,
                        class CudaNetwork **nets, size_t nnets,
                        LoadSensorsFunc load_sensors,
                        ProcessOutputFunc process_output);

        #define nbatches 3
        class CudaNetworkBatch *batches[nbatches];
        struct batch_bounds_t {
            size_t i;
            size_t n;
        } batch_bounds [nbatches];
    };

}
