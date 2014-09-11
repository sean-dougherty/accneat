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
	class Link {
	public: 
		double weight; // Weight of connection
        node_index_t in_node_index; // NNode inputting into the link

		Link(double w,
             node_index_t inode_index);
	};

} // namespace NEAT

#endif
