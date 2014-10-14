#pragma once

#include "networkmanager.h"

namespace NEAT {

    class CpuNetworkManager : public NetworkManager {
    public:
        virtual ~CpuNetworkManager() {}

        virtual std::unique_ptr<class Network> make_default() override;

        virtual void activate(class Network **nets, size_t nnets,
                              LoadSensorsFunc load_sensors,
                              ProcessOutputFunc process_output) override;
    };

}
