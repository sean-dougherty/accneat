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
#ifndef _GENOME_H_
#define _GENOME_H_

#include <vector>
#include "linkgene.h"
#include "nodegene.h"
#include "innovation.h"
#include "rng.h"

namespace NEAT {

	enum mutator {
		GAUSSIAN = 0,
		COLDGAUSSIAN = 1
	};

	class Genome {
        rng_t rng;
	public:
		int genome_id;

		std::vector<Trait> traits;
		std::vector<NodeGene> nodes;
		std::vector<LinkGene> links;

        void reset(int new_id);

		int get_last_node_id(); //Return id of final NodeGene in Genome
		double get_last_gene_innovnum(); //Return last innovation number in Genome

		void print_genome(); //Displays Genome on screen

        // todo: use c++11 move for constructor vectors?

        Genome();

		//Constructor which takes full genome specs and puts them into the new one
		Genome(int id,
               const std::vector<Trait> &t,
               const std::vector<NodeGene> &n,
               const std::vector<LinkGene> &g);

		//Special constructor which spawns off an input file
		//This constructor assumes that some routine has already read in GENOMESTART
        Genome(int id, std::ifstream &iFile);

		// Loads a new Genome from a file (doesn't require knowledge of Genome's id)
		static Genome *new_Genome_load(char *filename);

		//Destructor kills off all lists (including the trait vector)
		~Genome();

		// Dump this genome to specified file
		void print_to_file(std::ostream &outFile);
        void load_from_file(int id, std::istream &iFile);

        void duplicate_into(Genome &offspring, int new_id);

		// For debugging: A number of tests can be run on a genome to check its
		// integrity
		// Note: Some of these tests do not indicate a bug, but rather are meant
		// to be used to detect specific system states
		void verify();

		// ******* MUTATORS *******

		// Perturb params in one trait
		void mutate_random_trait();

		// Change random link's trait. Repeat times times
		void mutate_link_trait(int times);

		// Change random node's trait times times 
		void mutate_node_trait(int times);

		// Add Gaussian noise to linkweights either GAUSSIAN or COLDGAUSSIAN (from zero)
		void mutate_link_weights(double power,double rate,mutator mut_type);

		// toggle links on or off 
		void mutate_toggle_enable(int times);

		// Find first disabled gene and enable it 
		void mutate_gene_reenable();

		// These last kinds of mutations return false if they fail
		//   They can fail under certain conditions,  being unable
		//   to find a suitable place to make the mutation.
		//   Generally, if they fail, they can be called again if desired. 

		// Mutate genome by adding a node respresentation 
		bool mutate_add_node(int population_index,
                             std::vector<IndividualInnovation> &innovs);

		// Mutate the genome by adding a new link between 2 random NodeGenes 
		bool mutate_add_link(int population_index,
                             std::vector<IndividualInnovation> &innovs,
                             int tries); 

		// ****** MATING METHODS ***** 

		// This method mates this Genome with another Genome g.  
		//   For every point in each Genome, where each Genome shares
		//   the innovation number, the LinkGene is chosen randomly from 
		//   either parent.  If one parent has an innovation absent in 
		//   the other, the baby will inherit the innovation 
		//   Interspecies mating leads to all genes being inherited.
		//   Otherwise, excess genes come from most fit parent.
		void mate_multipoint(Genome *g,
                             Genome *offspring,
                             int genomeid,
                             double fitness1,
                             double fitness2);

		//This method mates like multipoint but instead of selecting one
		//   or the other when the innovation numbers match, it averages their
		//   weights 
		void mate_multipoint_avg(Genome *g,
                                 Genome *offspring,
                                 int genomeid,
                                 double fitness1,
                                 double fitness2);

		// ******** COMPATIBILITY CHECKING METHODS ********

		// This function gives a measure of compatibility between
		//   two Genomes by computing a linear combination of 3
		//   characterizing variables of their compatibilty.
		//   The 3 variables represent PERCENT DISJOINT GENES, 
		//   PERCENT EXCESS GENES, MUTATIONAL DIFFERENCE WITHIN
		//   MATCHING GENES.  So the formula for compatibility 
		//   is:  disjoint_coeff*pdg+excess_coeff*peg+mutdiff_coeff*mdmg.
		//   The 3 coefficients are global system parameters 
		double compatibility(Genome *g);

		double trait_compare(Trait *t1,Trait *t2);

		// Return number of non-disabled genes 
		int extrons();

		// Randomize the trait pointers of all the node and connection genes 
		void randomize_traits();

        Trait &get_trait(const NodeGene &node);
        Trait &get_trait(const LinkGene &gene);

	private:
        static bool nodelist_cmp(const NodeGene &a, const NodeGene &b) {
            return a.node_id < b.node_id;
        }
        static bool nodelist_cmp_key(const NodeGene &node, int node_id) {
            return node.node_id < node_id;
        }
        static bool linklist_cmp(const LinkGene &a, const LinkGene &b) {
            return a.innovation_num < b.innovation_num;
        }


		//Inserts a NodeGene into a given ordered list of NodeGenes in order
		static void add_node(std::vector<NodeGene> &nlist, const NodeGene &n);

		//Adds a new gene that has been created through a mutation in the
		//*correct order* into the list of links in the genome
		static void add_link(std::vector<LinkGene> &glist, const LinkGene &g);

    private:
        bool link_exists(int in_node_id, int out_node_id, bool is_recurrent);
        NodeGene *get_node(int id);
        
    private:
        class NodeLookup {
            std::vector<NodeGene> &nodes;
        public:
            // Must be sorted by node_id in ascending order
        NodeLookup(std::vector<NodeGene> &nodes_)
            : nodes(nodes_) {
            }

            NodeGene *find(int node_id) {
                auto it = std::lower_bound(nodes.begin(), nodes.end(), node_id, nodelist_cmp_key);
                if(it == nodes.end())
                    return nullptr;

                NodeGene &node = *it;
                if(node.node_id != node_id)
                    return nullptr;

                return &node;
            }

            NodeGene *find(NodeGene *n) {
                return find(n->node_id);
            }
        };

        class ProtoLinkGene {
            Genome *_genome = nullptr;
            //todo: does this have to be a LinkGene* now?
            LinkGene *_gene = nullptr;
            NodeGene *_in = nullptr;
            NodeGene *_out = nullptr;
        public:
            void set_gene(Genome *genome, LinkGene *gene) {
                _genome = genome;
                _gene = gene;
            }
            LinkGene *gene() {
                return _gene;
            }

            void set_out(NodeGene *out) {
                _out = out;
                _gene->set_out_node_id(out->node_id);
            }
            NodeGene *out() {
                return _out ? _out : _genome->get_node(_gene->out_node_id());
            }

            void set_in(NodeGene *in) {
                _in = in;
                _gene->set_in_node_id(in->node_id);
            }
            NodeGene *in() {
                return _in ? _in : _genome->get_node(_gene->in_node_id());
            }
        };
        NodeLookup node_lookup;
        Genome(const Genome &other);
    };

	void print_Genome_tofile(Genome *g,const char *filename);
} // namespace NEAT

#endif
