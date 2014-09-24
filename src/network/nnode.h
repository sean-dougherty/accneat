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

#include "link.h"
#include "neat.h"
#include "util.h"
#include <vector>

namespace NEAT {

    class Link;

	class NNode {
    public:
		real_t activation;
		real_t last_activation;
		nodetype type;
		std::vector<Link> incoming;

        NNode() {}
        NNode(nodetype type_)
            : activation(0)
            , last_activation(0)
            , type(type_) {
        }

        void flush() {
            switch(type) {
            case nodetype::BIAS:
            case nodetype::SENSOR:
                //no-op
                break;
            case nodetype::OUTPUT:
            case nodetype::HIDDEN:
                activation = 0.0;
                last_activation = 0.0;
                break;
            default:
                panic();
            }
        }

		// If the node is a SENSOR, returns true and loads the value
		void sensor_load(real_t value) {
            switch(type) {
            case nodetype::BIAS:
            case nodetype::SENSOR:
                last_activation = activation = value;
                break;
            default:
                panic();
            }
        }
	};


} // namespace NEAT

#endif
