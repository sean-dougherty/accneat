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
#ifndef _NNODE_H_
#define _NNODE_H_

#include <assert.h>
#include <algorithm>
#include <vector>
#include "neat.h"
#include "nodegene.h"
#include "trait.h"
#include "link.h"

namespace NEAT {

    class Link;

	class NNode {
    public:
		double activation;
		double last_activation;

	public:
		nodetype type; // type is either NEURON or SENSOR 
		std::vector<Link> incoming; // A list of pointers to incoming weighted signals from other nodes

		int node_id;
		nodeplace gen_node_label;

        NNode() {}
        NNode(NodeGene &gene);
		~NNode();

        void flush();

		// If the node is a SENSOR, returns true and loads the value
		void sensor_load(double);
	};


} // namespace NEAT

#endif
