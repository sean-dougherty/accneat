#pragma once

#include "neattypes.h"

namespace NEAT {

    class NetworkManager {
    public:
        //---
        //--- Types
        //---
        class BatchSensors {
        public:
            virtual void configure_step(size_t istep,
                                        const std::vector<real_t> &values,
                                        bool clear_noninput) = 0;
        };

        typedef std::function<void (class Network &net, size_t istep)> ProcessOutputFunc;

        //---
        //--- Static
        //---
        static NetworkManager *create();

        //---
        //--- Interface
        //---
        virtual ~NetworkManager() {}

        virtual std::unique_ptr<class Network> make_default() = 0;

        virtual std::unique_ptr<BatchSensors> make_batch_sensors(node_size_t nsensors,
                                                                 size_t nsteps) = 0;


        virtual void activate(class Network **nets,
                              size_t nnets,
                              BatchSensors *batch_sensors,
                              ProcessOutputFunc process_output) = 0;
                              
    };

}
