#pragma once

namespace NEAT {

    class NetworkManager {
    public:
        static NetworkManager *create();

        virtual ~NetworkManager() {}

        virtual std::unique_ptr<class Network> make_default() = 0;

        typedef std::function<bool (class Network &net, size_t istep)> LoadSensorsFunc;
        typedef std::function<void (class Network &net, size_t istep)> ProcessOutputFunc;

        virtual void activate(class Network **nets, size_t nnets,
                              LoadSensorsFunc load_sensors,
                              ProcessOutputFunc process_output) = 0;
    };

}
