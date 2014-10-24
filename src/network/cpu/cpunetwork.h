#pragma once

#include "network.h"

namespace NEAT {

    //---
    //--- CLASS CpuNetwork
    //---
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

        void clear_noninput();
        void load_sensor(size_t isensor, real_t activation);
        real_t *get_outputs();

        virtual void configure(const NetDims &dims,
                               NetNode *nodes,
                               NetLink *links);

        virtual NetDims get_dims() { return dims; }
	};

}
