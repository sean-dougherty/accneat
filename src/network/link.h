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

#include "neattypes.h"

namespace NEAT {

    typedef unsigned short node_index_t;

	// ----------------------------------------------------------------------- 
	// A LINK is a connection from one node to another with an associated weight 
	struct Link {
		real_t weight; // Weight of connection
        node_index_t in_node_index; // NNode inputting into the link
        node_index_t out_node_index; // NNode gaining energy from the link
	};

} // namespace NEAT
