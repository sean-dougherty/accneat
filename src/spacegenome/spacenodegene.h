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
            : type(type_)
            , location(location_) {
        }

        bool operator<(const SpaceNodeGene &other) const {
            return location < other.location;
        }

        friend std::ostream &operator<<(std::ostream &out, const SpaceNodeGene &node) {
            return out << "node "
                       << node.location << " "
                       << node.trait_id << " "
                       << (int)node.type;
        }
	};

} // namespace NEAT

