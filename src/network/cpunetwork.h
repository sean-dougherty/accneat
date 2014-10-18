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
        std::vector<real_t> &get_activations(__out std::vector<real_t> &result);
        void set_activations(__in std::vector<real_t> &newacts);
        void load_sensors(const std::vector<real_t> &sensvals,
                          size_t off,
                          bool clear_noninput);

        virtual void configure(const NetDims &dims,
                               NetNode *nodes,
                               NetLink *links) override;

        virtual NetDims get_dims() override { return dims; }

        virtual real_t get_output(size_t index) override;
	};

}
