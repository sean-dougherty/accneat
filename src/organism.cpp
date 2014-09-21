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

#include "organism.h"
#include "species.h"

using namespace NEAT;
using std::vector;

Organism::Organism(const Organism &other) {
    this->genome = other.genome->make_default();
    other.copy_into(*this);
}

Organism::Organism(const Genome &genome) {
    this->genome = genome.make_clone();

    //Note: We're in the base class constructor, so a derived class' init() won't
    //      be called. The derived class' constructor must also call init().
    init(0);
}

Organism::~Organism() {
}

void Organism::init(int gen) {
	fitness=0.0;
	generation=gen;
	error=0;
}

void Organism::create_phenotype() {
	real_t maxweight=0.0; //Compute the maximum weight for adaptation purposes
	real_t weight_mag; //Measures absolute value of weights

    net.reset();
    vector<NNode> &netnodes = net.nodes;

	//Create the nodes
	for(NodeGene &node: genome->nodes) {
        netnodes.emplace_back(node);
	}

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
    } node_lookup(netnodes);

	//Create the links by iterating through the genes
    for(LinkGene &gene: genome->links) {
		//Only create the link if the gene is enabled
		if(gene.enable) {
            node_index_t inode = node_lookup.find(gene.in_node_id());
            node_index_t onode = node_lookup.find(gene.out_node_id());

			//NOTE: This line could be run through a recurrency check if desired
			// (no need to in the current implementation of NEAT)
			netnodes[onode].incoming.emplace_back(gene.weight(), inode);

            Link &newlink = netnodes[onode].incoming.back();

			//Keep track of maximum weight
			if (newlink.weight>0)
				weight_mag=newlink.weight;
			else weight_mag=-newlink.weight;
			if (weight_mag>maxweight)
				maxweight=weight_mag;
		}
	}

    net.init(maxweight);
}

Organism &Organism::operator=(const Organism &other) {
    other.copy_into(*this);
    return *this;
}

void Organism::copy_into(Organism &dst) const {
#define copy(field) dst.field = this->field;
    
    copy(population_index);
    copy(fitness);
    copy(error);
    copy(net);
    *dst.genome = *this->genome;
    copy(generation);

#undef copy
}
