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

    #define NODES_MAX USHRT_MAX
    #define LINKS_MAX USHRT_MAX

    typedef unsigned short node_size_t;
    typedef unsigned short link_size_t;

    struct NetDims {
        struct {
            node_size_t bias;
            node_size_t sensor;
            node_size_t output;
            node_size_t hidden;

            node_size_t all;
            node_size_t input;
            node_size_t noninput;
        } nnodes;
        
        link_size_t nlinks;
    };

	struct NetLink {
		real_t weight; // Weight of connection
        node_size_t in_node_index; // NetNode inputting into the link
        node_size_t out_node_index; // NetNode gaining energy from the link
	};

	struct NetNode {
        link_size_t incoming_start;
        link_size_t incoming_end;
	};

	class Network {
    public:
        size_t population_index;

		virtual ~Network() {}

        virtual void configure(const NetDims &dims,
                               NetNode *nodes,
                               NetLink *links) = 0;

        virtual NetDims get_dims() = 0;

		// Takes an array of sensor values and loads it into SENSOR inputs ONLY
		virtual void load_sensors(const std::vector<real_t> &sensvals,
                                  bool clear_noninput) = 0;

        virtual real_t get_output(size_t index) = 0;
	};

} // namespace NEAT

#endif
