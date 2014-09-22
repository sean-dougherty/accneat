#pragma once

#include "innovnodegene.h"
#include <vector>

namespace NEAT {

    inline bool nodelist_cmp(const InnovNodeGene &a, const InnovNodeGene &b) {
        return a.node_id < b.node_id;
    }
    inline bool nodelist_cmp_key(const InnovNodeGene &node, int node_id) {
        return node.node_id < node_id;
    }

    class InnovNodeLookup {
        std::vector<InnovNodeGene> &nodes;
    public:
        // Must be sorted by node_id in ascending order
    InnovNodeLookup(std::vector<InnovNodeGene> &nodes_)
        : nodes(nodes_) {
        }

        InnovNodeGene *find(int node_id) {
            auto it = std::lower_bound(nodes.begin(), nodes.end(), node_id, nodelist_cmp_key);
            if(it == nodes.end())
                return nullptr;

            InnovNodeGene &node = *it;
            if(node.node_id != node_id)
                return nullptr;

            return &node;
        }

        InnovNodeGene *find(InnovNodeGene *n) {
            return find(n->node_id);
        }
    };

}
