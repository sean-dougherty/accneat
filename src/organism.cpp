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

Organism::Organism() {
    init(0, nullptr);
}

void Organism::init(int gen, const char* md) {
	fitness=0.0;
	orig_fitness=0.0;
	species=0;  //Start it in no Species
	expected_offspring=0;
	generation=gen;
	eliminate=false;
	error=0;
	winner=false;
	champion=false;
	super_champ_offspring=0;

	// If md is null, then we don't have metadata, otherwise we do have metadata so copy it over
	if(md == 0) {
		strcpy(metadata, "");
	} else {
		strncpy(metadata, md, 128);
	}

	time_alive=0;

	modified = true;
}

Organism::~Organism() {
}

void Organism::create_phenotype() {
	real_t maxweight=0.0; //Compute the maximum weight for adaptation purposes
	real_t weight_mag; //Measures absolute value of weights

    net.reset();
    vector<NNode> &netnodes = net.nodes;

	//Create the nodes
	for(NodeGene &node: genome.nodes) {
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
    for(LinkGene &gene: genome.links) {
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

void Organism::write(std::ostream &outFile) {
	
	char tempbuf2[1024];
	if(modified) {
		sprintf(tempbuf2, "/* Organism #%d Fitness: %f Time: %d */\n", genome.genome_id, fitness, time_alive);
	} else {
		sprintf(tempbuf2, "/* %s */\n", metadata);
	}
	outFile << tempbuf2;
	genome.print(outFile);
}
bool NEAT::order_orgs(Organism *x, Organism *y) {
	return (x)->fitness > (y)->fitness;
}

bool NEAT::order_orgs_by_adjusted_fit(Organism *x, Organism *y) {
	return (x)->fitness / (x->species)->organisms.size()  > (y)->fitness / (y->species)->organisms.size();
}
