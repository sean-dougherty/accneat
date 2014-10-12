#pragma once

#include "network.h"

namespace NEAT {

	class CpuNetwork : public Network {
    private:
        NodeCounts counts;
		std::vector<NetNode> nodes;
		std::vector<NetLink> links;
        std::vector<real_t> activations;

    public:
        CpuNetwork() {}
		virtual ~CpuNetwork() {}

        virtual Network &operator=(const Network &other) override;

        virtual void configure(const NodeCounts &counts,
                               NetNode *nodes, node_size_t nnodes,
                               NetLink *links, link_size_t nlinks) override;

		// Puts the network back into an inactive state
		virtual void flush() override;
		
		virtual void activate(size_t ncycles) override;

		// Takes an array of sensor values and loads it into SENSOR inputs ONLY
		virtual void load_sensors(const std::vector<real_t> &sensvals) override;
        virtual real_t get_output(size_t index) override;
	};

}
