#pragma once

#include "nnode.h"
#include <algorithm>
#include <vector>

namespace NEAT {

    class NetNodeLookup {
        std::vector<NNode> &nodes;

        static bool cmp(const NNode &node, int node_id) {
            return node.node_id < node_id;
        }
    public:
        // Must be sorted by node_id in ascending order
        NetNodeLookup(std::vector<NNode> &nodes_)
            : nodes(nodes_) {
        }

        node_index_t find(int node_id) {
            auto it = std::lower_bound(nodes.begin(), nodes.end(), node_id, cmp);
            assert(it != nodes.end());

            node_index_t i = it - nodes.begin();
            assert(nodes[i].node_id == node_id);

            return i;
        }
    };

}
