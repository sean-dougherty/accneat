#pragma once

#include "innovnodegene.h"
#include <vector>

namespace NEAT {
    inline bool nodelist_cmp_key(const SpaceNodeGene &node, const NodeLocation &location) {
        return node.location < location;
    }

    class SpaceNodeLookup {
        std::vector<SpaceNodeGene> &nodes;
    public:
        // Must be sorted by location in ascending order
    SpaceNodeLookup(std::vector<SpaceNodeGene> &nodes_)
        : nodes(nodes_) {
        }

        SpaceNodeGene *find(const NodeLocation &location) {
            auto it = std::lower_bound(nodes.begin(), nodes.end(), location, nodelist_cmp_key);
            if(it == nodes.end())
                return nullptr;

            SpaceNodeGene &node = *it;
            if(node.location != location)
                return nullptr;

            return &node;
        }

        SpaceNodeGene *find(SpaceNodeGene *n) {
            return find(n->location);
        }
    };

}
