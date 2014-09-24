#pragma once

#include "neat.h"
#include "nodelocation.h"
#include "trait.h"
#include "link.h"
#include "network.h"

namespace NEAT {

	class SpaceLinkGene {
	public:
		int trait_id;  // identify the trait derived by this link
		real_t weight; // Weight of connection
		NodeLocation in_node_loc; // NNode inputting into the link
		NodeLocation out_node_loc; // NNode that the link affects

        static SpaceLinkGene create_search_key(const NodeLocation &in_node_loc_,
                                               const NodeLocation &out_node_loc_) {
            return SpaceLinkGene(0, 0, in_node_loc_, out_node_loc_);
        }

        // Construct a gene in an invalid default state.
        SpaceLinkGene() {}

		//Construct a gene with a trait
		SpaceLinkGene(int trait_id_,
                      real_t weight_,
                      const NodeLocation &in_node_loc_,
                      const NodeLocation &out_node_loc_)
            : trait_id(trait_id_)
            , weight(weight_)
            , in_node_loc(in_node_loc_)
            , out_node_loc(out_node_loc_) {
        }

        bool operator<(const SpaceLinkGene &other) const {
            if(in_node_loc < other.in_node_loc)
                return true;
            if((in_node_loc == other.in_node_loc) && (out_node_loc < other.out_node_loc))
                return true;
            return false;
        }

        friend std::ostream &operator<<(std::ostream &out, const SpaceLinkGene &link) {
            return out << "link "
                       << link.trait_id << " "
                       << link.in_node_loc << " "
                       << link.out_node_loc << " "
                       << link.weight << " ";
        }
	};

} // namespace NEAT

