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
#pragma once

#include <assert.h>
#include "neat.h"

namespace NEAT {

	class InnovNodeGene {
		int trait_id;  // identify the trait derived by this node
	public:
		bool frozen; // When frozen, cannot be mutated (meaning its trait pointer is fixed)
		nodetype type;
		int node_id;  // A node can be given an identification number for saving in files

        // Construct InnovNodeGene with invalid state.
        InnovNodeGene() {}
		InnovNodeGene(nodetype ntype,int nodeid);
		// Construct the node out of a file specification using given list of traits
		InnovNodeGene (const char *argline);

		~InnovNodeGene();

        inline void set_trait_id(int id) { assert(id > 0); trait_id = id; }
        inline int get_trait_id() const {return trait_id;}

		inline const nodetype get_type() const {return type;}
		inline void set_type(nodetype t) {type = t;}

		// Print the node to a file
        void  print_to_file(std::ostream &outFile);
	};

} // namespace NEAT

