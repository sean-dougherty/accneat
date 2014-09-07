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
#include "gene.h"
#include "innovation.h"

namespace NEAT {

	enum mutator {
		GAUSSIAN = 0,
		COLDGAUSSIAN = 1
	};

	//----------------------------------------------------------------------- 
	//A Genome is the primary source of genotype information used to create   
	//a phenotype.  It contains 3 major constituents:                         
	//  1) A list of Traits                                                 
	//  2) A list of NNodes pointing to a Trait from (1)                      
	//  3) A list of Genes with Links that point to Traits from (1)           
	//(1) Reserved parameter space for future use
	//(2) NNode specifications                                                
	//(3) Is the primary source of innovation in the evolutionary Genome.     
	//    Each Gene in (3) has a marker telling when it arose historically.   
	//    Thus, these Genes can be used to speciate the population, and the   
	//    list of Genes provide an evolutionary history of innovation and     
	//    link-building.

	class Genome {
	public:
		int genome_id;

		std::vector<Trait> traits; //parameter conglomerations
		std::vector<NNode*> nodes; //List of NNodes for the Network
		std::vector<Gene*> genes; //List of innovation-tracking genes

		Network *phenotype; //Allows Genome to be matched with its Network

		int get_last_node_id(); //Return id of final NNode in Genome
		double get_last_gene_innovnum(); //Return last innovation number in Genome

		void print_genome(); //Displays Genome on screen

        // todo: use c++11 move for constructor vectors?

		//Constructor which takes full genome specs and puts them into the new one
		Genome(int id,
               const std::vector<Trait> &t,
               std::vector<NNode*> n,
               std::vector<Gene*> g);

		//Special constructor which spawns off an input file
		//This constructor assumes that some routine has already read in GENOMESTART
        Genome(int id, std::ifstream &iFile);

		// Loads a new Genome from a file (doesn't require knowledge of Genome's id)
		static Genome *new_Genome_load(char *filename);

		//Destructor kills off all lists (including the trait vector)
		~Genome();

		//Generate a network phenotype from this Genome with specified id
		Network *genesis(int);

		// Dump this genome to specified file
		void print_to_file(std::ostream &outFile);
		void print_to_file(std::ofstream &outFile);

		// Wrapper for print_to_file above
		void print_to_filename(char *filename);

		// Duplicate this Genome to create a new one with the specified id 
		Genome *duplicate(int new_id);

		// For debugging: A number of tests can be run on a genome to check its
		// integrity
		// Note: Some of these tests do not indicate a bug, but rather are meant
		// to be used to detect specific system states
		bool verify();

		// ******* MUTATORS *******

		// Perturb params in one trait
		void mutate_random_trait();

		// Change random link's trait. Repeat times times
		void mutate_link_trait(int times);

		// Change random node's trait times times 
		void mutate_node_trait(int times);

		// Add Gaussian noise to linkweights either GAUSSIAN or COLDGAUSSIAN (from zero)
		void mutate_link_weights(double power,double rate,mutator mut_type);

		// toggle genes on or off 
		void mutate_toggle_enable(int times);

		// Find first disabled gene and enable it 
		void mutate_gene_reenable();

		// These last kinds of mutations return false if they fail
		//   They can fail under certain conditions,  being unable
		//   to find a suitable place to make the mutation.
		//   Generally, if they fail, they can be called again if desired. 

		// Mutate genome by adding a node respresentation 
		bool mutate_add_node(std::vector<Innovation*> &innovs,int &curnode_id,double &curinnov);

		// Mutate the genome by adding a new link between 2 random NNodes 
		bool mutate_add_link(std::vector<Innovation*> &innovs,double &curinnov,int tries); 

		// ****** MATING METHODS ***** 

		// This method mates this Genome with another Genome g.  
		//   For every point in each Genome, where each Genome shares
		//   the innovation number, the Gene is chosen randomly from 
		//   either parent.  If one parent has an innovation absent in 
		//   the other, the baby will inherit the innovation 
		//   Interspecies mating leads to all genes being inherited.
		//   Otherwise, excess genes come from most fit parent.
		Genome *mate_multipoint(Genome *g,int genomeid,double fitness1, double fitness2, bool interspec_flag);

		//This method mates like multipoint but instead of selecting one
		//   or the other when the innovation numbers match, it averages their
		//   weights 
		Genome *mate_multipoint_avg(Genome *g,int genomeid,double fitness1,double fitness2, bool interspec_flag);

		// This method is similar to a standard single point CROSSOVER
		//   operator.  Traits are averaged as in the previous 2 mating
		//   methods.  A point is chosen in the smaller Genome for crossing
		//   with the bigger one.  
		Genome *mate_singlepoint(Genome *g,int genomeid);


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

	protected:
		//Inserts a NNode into a given ordered list of NNodes in order
		void node_insert(std::vector<NNode*> &nlist, NNode *n);

		//Adds a new gene that has been created through a mutation in the
		//*correct order* into the list of genes in the genome
		void add_gene(std::vector<Gene*> &glist,Gene *g);

    private:
        Trait &get_trait(NNode *node);
        Trait &get_trait(Gene *gene);
        bool link_exists(int in_node_id, int out_node_id, bool is_recurrent);
        NNode *get_node(int id);
        
    private:
        class NodeLookup {
            std::vector<NNode *> &nodes;

            static bool cmp(NNode *node, int node_id) {
                return node->node_id < node_id;
            }
        public:
            // Must be sorted by node_id in ascending order
        NodeLookup(std::vector<NNode *> &nodes_)
            : nodes(nodes_) {
            }

            NNode *find(int node_id) {
                auto it = std::lower_bound(nodes.begin(), nodes.end(), node_id, cmp);
                if(it == nodes.end())
                    return nullptr;

                NNode *node = *it;
                if(node->node_id != node_id)
                    return nullptr;

                return node;
            }

            NNode *find(NNode *n) {
                return find(n->node_id);
            }
        };

        class ProtoGene {
            Genome *_genome = nullptr;
            Gene *_gene = nullptr;
            NNode *_in = nullptr;
            NNode *_out = nullptr;
        public:
            void set_gene(Genome *genome, Gene *gene) {
                _genome = genome;
                _gene = gene;
            }
            Gene *gene() {
                return _gene;
            }

            void set_out(NNode *out) {
                _out = out;
                _gene->lnk->out_node = out; // tmp
            }
            NNode *out() {
                return _out ? _out : _genome->get_node(_gene->out_node_id());
            }

            void set_in(NNode *in) {
                _in = in;
                _gene->lnk->in_node = in; // tmp
            }
            NNode *in() {
                return _in ? _in : _genome->get_node(_gene->in_node_id());
            }
        };
        NodeLookup node_lookup;
        Genome(const Genome &other);
    };

	void print_Genome_tofile(Genome *g,const char *filename);
} // namespace NEAT

#endif
