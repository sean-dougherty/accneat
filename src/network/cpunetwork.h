#pragma once

#include "network.h"

namespace NEAT {

	class CpuNetwork : public Network {
    private:
        NetDims dims;
		std::vector<NetNode> nodes;
		std::vector<NetLink> links;
        std::vector<real_t> activations;

    public:
        CpuNetwork() {}
		virtual ~CpuNetwork() {}

		void activate(size_t ncycles);

        virtual void configure(const NetDims &dims,
                               NetNode *nodes,
                               NetLink *links) override;


		// Takes an array of sensor values and loads it into SENSOR inputs ONLY
		virtual void load_sensors(const std::vector<real_t> &sensvals,
                                  bool clear_noninput) override;

        virtual real_t get_output(size_t index) override;
	};

}
