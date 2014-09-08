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
#ifndef _LINK_H_
#define _LINK_H_

#include "neat.h"
#include "trait.h"
#include "nnode.h"

namespace NEAT {

	class NNode;

    typedef uint16_t node_index_t;

	// ----------------------------------------------------------------------- 
	// A LINK is a connection from one node to another with an associated weight 
	// It can be marked as recurrent 
	// Its parameters are made public for efficiency 
	class Link {
	public: 
        node_index_t in_node_index; // NNode inputting into the link
        node_index_t out_node_index; // NNode that the link affects

		double weight; // Weight of connection
		bool is_recurrent;
		bool time_delay;


		int trait_id;  // identify the trait derived by this link

		// ************ LEARNING PARAMETERS *********** 
		// These are link-related parameters that change during Hebbian type learning

		double added_weight;  // The amount of weight adjustment 
		double params[NEAT::num_trait_params];

		// Including a trait in the Link creation
		Link(int trait_id,
             double w,
             node_index_t inode_index,
             node_index_t onode_index,
             bool recur);

		Link(double w,
             node_index_t inode_index,
             node_index_t onode_index,
             bool recur);

		// For when you don't know the connections yet
		Link(double w);

        inline int get_trait_id() {
            return trait_id;
        }
        inline void set_trait_id(int trait_id_) {trait_id = trait_id_;}

		// Derive a trait into link params
		void derive_trait(const Trait &curtrait);

	};

} // namespace NEAT

#endif
