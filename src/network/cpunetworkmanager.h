#pragma once

#include "networkmanager.h"

namespace NEAT {

    class CpuNetworkManager : public NetworkManager {
    public:
        virtual ~CpuNetworkManager() {}

        virtual std::unique_ptr<class Network> make_default() override;

        virtual std::unique_ptr<BatchSensors> make_batch_sensors(node_size_t nsensors,
                                                                 size_t nsteps) override;

        virtual void activate(class Network **nets,
                              size_t nnets,
                              BatchSensors *batch_sensors,
                              ProcessOutputFunc process_output) override;
    };

}
