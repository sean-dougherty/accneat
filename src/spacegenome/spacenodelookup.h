#pragma once

#include "innovnodegene.h"

namespace NEAT {
    class SpaceNodeLookup {
        std::vector<SpaceNodeGene> &nodes;
    public:
        // Must be sorted by location in ascending order
    SpaceNodeLookup(std::vector<SpaceNodeGene> &nodes_)
        : nodes(nodes_) {
        }

        SpaceNodeGene *find(const SpaceNodeGene *n) {
            auto it = std::lower_bound(nodes.begin(), nodes.end(), *n);
            if(it == nodes.end())
                return nullptr;

            SpaceNodeGene &node = *it;
            if(node.location != n->location)
                return nullptr;

            return &node;
        }

        SpaceNodeGene *find(const NodeLocation &location) {
            for(nodetype type: nodetypes) {
                SpaceNodeGene key{type, location};
                SpaceNodeGene *result = find(&key);
                if(result)
                    return result;
            }
            return nullptr;
        }
    };

}
