/*
 Copyright 2001 The University of Texas at Austin

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/
#ifndef _NETWORK_H_
#define _NETWORK_H_

#include "neattypes.h"

namespace NEAT {

    struct NodeCounts {
        size_t nnodes;
        size_t nbias_nodes;
        size_t nsensor_nodes;
        size_t ninput_nodes;
        size_t noutput_nodes;
        size_t nhidden_nodes;
    };

    typedef unsigned short node_index_t;

	// ----------------------------------------------------------------------- 
	// A LINK is a connection from one node to another with an associated weight 
	struct Link {
		real_t weight; // Weight of connection
        node_index_t in_node_index; // NNode inputting into the link
        node_index_t out_node_index; // NNode gaining energy from the link
	};

    typedef unsigned short link_index_t;

	struct NNode {
        link_index_t incoming_start;
        link_index_t incoming_end;
	};

	// ----------------------------------------------------------------------- 
	// A NETWORK is a LIST of input NODEs and a LIST of output NODEs           
	//   The point of the network is to define a single entity which can evolve
	//   or learn on its own, even though it may be part of a larger framework 
	class Network {
    private:
        NodeCounts counts;
		std::vector<NNode> nodes;
		std::vector<Link> links;

        std::vector<real_t> activations_buffers[2];
        real_t *activations, *last_activations;
    public:
        Network();
		~Network();

        void init(const NodeCounts &counts,
                  NNode *nodes, size_t nnodes,
                  Link *links, size_t nlinks);

		// Puts the network back into an inactive state
		void flush();
		
		// Activates the net such that all outputs are active
		void activate();

		// Takes an array of sensor values and loads it into SENSOR inputs ONLY
		void load_sensors(const real_t*);
		void load_sensors(const std::vector<real_t> &sensvals);

        real_t get_output(size_t index);
	};

} // namespace NEAT

#endif
