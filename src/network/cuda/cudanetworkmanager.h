#pragma once

namespace NEAT {

#if false
    class CudaNetworkManager : public NetworkManager {
    public:
        CudaNetworkManager();
        virtual ~CudaNetworkManager() {}

        virtual std::unique_ptr<class Network> make_default() override;

        virtual std::unique_ptr<BatchSensors> make_batch_sensors(node_size_t nsensors,
                                                                 size_t nsteps) override;

        virtual void activate(class Network **nets,
                              size_t nnets,
                              BatchSensors *batch_sensors,
                              ProcessOutputFunc process_output) override;

    private:
        void __activate(size_t ibatch,
                        class CudaNetwork **nets, size_t nnets,
                        class CudaBatchSensors *batch_sensors,
                        ProcessOutputFunc process_output);

        #define nbatches 3
        class CudaNetworkBatch *batches[nbatches];
        struct batch_bounds_t {
            size_t i;
            size_t n;
        } batch_bounds [nbatches];
    };

#endif // false
}
