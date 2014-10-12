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
#pragma once

#include "genome.h"
#include "innovlinkgene.h"
#include "innovnodegene.h"
#include "innovnodelookup.h"
#include "innovation.h"

namespace NEAT {

	class InnovGenome : public Genome {
	public:
		std::vector<Trait> traits;
		std::vector<InnovNodeGene> nodes;
		std::vector<InnovLinkGene> links;

		int get_last_node_id(); //Return id of final InnovNodeGene in InnovGenome
		real_t get_last_gene_innovnum(); //Return last innovation number in InnovGenome

        InnovGenome();
        InnovGenome(rng_t rng,
                    size_t ntraits,
                    size_t ninputs,
                    size_t noutputs,
                    size_t nhidden);

        virtual Genome &operator=(const Genome &other) override;

		//Destructor kills off all lists (including the trait vector)
		virtual ~InnovGenome();

		// Dump this genome to specified file
		virtual void print(std::ostream &out) override;

        void duplicate_into(InnovGenome *offspring) const;
        InnovGenome &operator=(const InnovGenome &other);

		// For debugging: A number of tests can be run on a genome to check its
		// integrity
		// Note: Some of these tests do not indicate a bug, but rather are meant
		// to be used to detect specific system states
		virtual void verify() override;
        virtual Stats get_stats() override;

		// ******* MUTATORS *******

		// Perturb params in one trait
		void mutate_random_trait();

		// Change random link's trait. Repeat times times
		void mutate_link_trait(int times);

		// Change random node's trait times times 
		void mutate_node_trait(int times);

		// Add Gaussian noise to linkweights either GAUSSIAN or COLDGAUSSIAN (from zero)
		void mutate_link_weights(real_t power,real_t rate,mutator mut_type);

		// toggle links on or off 
		void mutate_toggle_enable(int times);

		// Find first disabled gene and enable it 
		void mutate_gene_reenable();

		// These last kinds of mutations return false if they fail
		//   They can fail under certain conditions,  being unable
		//   to find a suitable place to make the mutation.
		//   Generally, if they fail, they can be called again if desired. 

		// Mutate genome by adding a node respresentation 
		bool mutate_add_node(CreateInnovationFunc create_innov,
                             bool delete_split_link);

		void mutate_delete_node();

		void mutate_delete_link();

		// Mutate the genome by adding a new link between 2 random InnovNodeGenes 
		bool mutate_add_link(CreateInnovationFunc create_innov,
                             int tries); 

		// ****** MATING METHODS ***** 
		static void mate(InnovGenome *genome1,
                         InnovGenome *genome2,
                         InnovGenome *offspring,
                         real_t fitness1,
                         real_t fitness2);

		//   For every point in each InnovGenome, where each InnovGenome shares
		//   the innovation number, the InnovLinkGene is chosen randomly from 
		//   either parent.  If one parent has an innovation absent in 
		//   the other, the baby will inherit the innovation 
		//   Interspecies mating leads to all genes being inherited.
		//   Otherwise, excess genes come from most fit parent.
		static void mate_multipoint(InnovGenome *genome1,
                                    InnovGenome *genome2,
                                    InnovGenome *offspring,
                                    real_t fitness1,
                                    real_t fitness2);

		//This method mates like multipoint but instead of selecting one
		//   or the other when the innovation numbers match, it averages their
		//   weights 
		static void mate_multipoint_avg(InnovGenome *genome1,
                                        InnovGenome *genome2,
                                        InnovGenome *offspring,
                                        real_t fitness1,
                                        real_t fitness2);

		// ******** COMPATIBILITY CHECKING METHODS ********

		// This function gives a measure of compatibility between
		//   two InnovGenomes by computing a linear combination of 3
		//   characterizing variables of their compatibilty.
		//   The 3 variables represent PERCENT DISJOINT GENES, 
		//   PERCENT EXCESS GENES, MUTATIONAL DIFFERENCE WITHIN
		//   MATCHING GENES.  So the formula for compatibility 
		//   is:  disjoint_coeff*pdg+excess_coeff*peg+mutdiff_coeff*mdmg.
		//   The 3 coefficients are global system parameters 
		real_t compatibility(InnovGenome *g);

		real_t trait_compare(Trait *t1,Trait *t2);

		// Randomize the trait pointers of all the node and connection genes 
		void randomize_traits();

        Trait &get_trait(const InnovNodeGene &node);
        Trait &get_trait(const InnovLinkGene &gene);

        InnovNodeGene *get_node(int id);
        node_size_t get_node_index(int id);

        virtual void init_phenotype(class Network &net) override;

	private:
        void reset();

        static bool linklist_cmp(const InnovLinkGene &a, const InnovLinkGene &b) {
            return a.innovation_num < b.innovation_num;
        }

		//Inserts a InnovNodeGene into a given ordered list of InnovNodeGenes in order
		static void add_node(std::vector<InnovNodeGene> &nlist, const InnovNodeGene &n);

		//Adds a new gene that has been created through a mutation in the
		//*correct order* into the list of links in the genome
		static void add_link(std::vector<InnovLinkGene> &glist, const InnovLinkGene &g);

    private:
        InnovLinkGene *find_link(int in_node_id, int out_node_id, bool is_recurrent);
        void delete_if_orphaned_hidden_node(int node_id);
        void delete_link(InnovLinkGene *link);

        InnovNodeLookup node_lookup;
    };
}

