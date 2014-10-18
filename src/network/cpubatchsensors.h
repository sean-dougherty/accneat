#pragma once

#include "cpunetwork.h"
#include "networkmanager.h"
#include <assert.h>

namespace NEAT {

    class CpuBatchSensors : public NetworkManager::BatchSensors {
        node_size_t nsensors;
        size_t nsteps;
        std::vector<real_t> sensor_vals;
        std::vector<bool> clear_noninput;

        inline size_t sensors_start(size_t step) {return step * nsensors;}

    public:
        CpuBatchSensors(node_size_t nsensors_,
                        size_t nsteps_)
            : nsensors(nsensors_)
            , nsteps(nsteps_) {

            sensor_vals.resize(sensors_start(nsteps));
            clear_noninput.resize(nsteps);
        }

        size_t get_nsteps() {return nsteps;}

        void load_sensors(CpuNetwork &net, size_t istep) {
            net.load_sensors(sensor_vals, sensors_start(istep), clear_noninput[istep]);
        }

        virtual void configure_step(size_t istep,
                                    const std::vector<real_t> &values,
                                    bool clear_noninput_) override {
            assert(istep < nsteps);
            assert(values.size() == nsensors);

            for(size_t i = 0, off = sensors_start(istep); i < nsensors; i++) {
                sensor_vals[off + i] = values[i];
            }
            clear_noninput[istep] = clear_noninput_;
        }
    };

}
