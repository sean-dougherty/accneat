#pragma once

#include "neat.h"
#include "nodelocation.h"
#include <assert.h>
#include <iostream>

namespace NEAT {

	class SpaceNodeGene {
	public:
		int trait_id;  // identify the trait derived by this node
		nodetype type;
        NodeLocation location;
        
        // Construct SpaceNodeGene with invalid state.
        SpaceNodeGene() {}
		SpaceNodeGene(nodetype type_,
                      NodeLocation location_)
            : trait_id(1)
            , type(type_)
            , location(location_) {
        }

        bool operator<(const SpaceNodeGene &other) const {
            if(type == other.type) {
                return location < other.location;
            } else {
                return type < other.type;
            }
        }

        friend std::ostream &operator<<(std::ostream &out, const SpaceNodeGene &node) {
            return out << "node "
                       << node.trait_id << " "
                       << (int)node.type << " "
                       << node.location;
        }
	};

} // namespace NEAT

