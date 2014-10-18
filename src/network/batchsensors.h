#pragma once

#include "neattypes.h"

namespace NEAT {

    class BatchSensors {
    public:
        virtual ~BatchSensors() {}

        virtual void configure_step(size_t istep,
                                    const std::vector<real_t> &values,
                                    bool clear_noninput) = 0;
    };

}
