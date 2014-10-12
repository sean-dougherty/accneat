#pragma once

namespace NEAT {

    class NetworkManager {
    public:
        NetworkManager() {}
        virtual ~NetworkManager() {}

        virtual std::unique_ptr<class Network> make_default();
    };

}
