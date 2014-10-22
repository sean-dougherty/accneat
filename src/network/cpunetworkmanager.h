#pragma once

namespace NEAT {

    class CpuNetworkManager : public NetworkManager {
    public:
        virtual ~CpuNetworkManager() {}

        virtual std::unique_ptr<class Network> make_default() override;
    };

}
