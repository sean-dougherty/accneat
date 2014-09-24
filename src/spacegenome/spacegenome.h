#pragma once

#include "genome.h"
#include "spacelinkgene.h"
#include "spacenodegene.h"
#include "spacenodelookup.h"
#include <vector>

namespace NEAT {

	class SpaceGenome : public Genome {
	public:
		std::vector<Trait> traits;
		std::vector<SpaceNodeGene> nodes;
		std::vector<SpaceLinkGene> links;

        SpaceGenome()
            : node_lookup(nodes) {
        }

        SpaceGenome(rng_t rng,
                    size_t ntraits,
                    size_t ninputs,
                    size_t noutputs,
                    size_t nhidden);

        void duplicate_into(SpaceGenome *offspring);
        SpaceGenome &operator=(const SpaceGenome &other);

        virtual std::unique_ptr<Genome> make_default() const override;
        virtual std::unique_ptr<Genome> make_clone() const override;

		// Dump this genome to specified file
		virtual void print(std::ostream &out) override;

		// For debugging: A number of tests can be run on a genome to check its
		// integrity
		// Note: Some of these tests do not indicate a bug, but rather are meant
		// to be used to detect specific system states
		virtual void verify() override;
        virtual Stats get_stats() override;

        virtual void init_phenotype(class Network &net) override;

        void mutate();
		void randomize_traits();
		void mutate_link_weights(real_t power,
                                 real_t rate,
                                 mutator mut_type);
		void mutate_add_link(); 

		static void mate(SpaceGenome *genome1,
                         SpaceGenome *genome2,
                         SpaceGenome *offspring,
                         real_t fitness1,
                         real_t fitness2);

    private:
        void reset();
		void mutate_random_trait();
		void mutate_link_trait(int times = 1);
		void mutate_node_trait(int times = 1);
		void mutate_add_node();
        // todo: mutate_move_node()
		void mutate_delete_node();
		void mutate_delete_link();

		static void mate_singlepoint(SpaceGenome *genome1,
                                     SpaceGenome *genome2,
                                     SpaceGenome *offspring);

        void add_link(const SpaceLinkGene &link);
        void delete_link(const SpaceLinkGene &link);
        SpaceLinkGene *find_link(const NodeLocation &in_node_loc,
                                 const NodeLocation &out_node_loc);
        void add_node(const SpaceNodeGene &node);
        SpaceNodeGene *get_node(const NodeLocation &loc);
        node_index_t get_node_index(const NodeLocation &loc);
        bool create_random_node_location(const NodeLocation &loc1,
                                         const NodeLocation &loc2,
                                         NodeLocation &result,
                                         bool empty_space_required = true,
                                         int maxtries = 10);
        bool create_random_node_location(const NodeLocation &center,
                                         NodeLocation &result,
                                         bool empty_space_required = true,
                                         int maxtries = 10);
        void delete_if_orphaned_hidden_node(const NodeLocation &loc);

        SpaceNodeLookup node_lookup;

/*
        InnovGenome(rng_t rng,
                    size_t ntraits,
                    size_t ninputs,
                    size_t noutputs,
                    size_t nhidden);
		//Constructor which takes full genome specs and puts them into the new one
		InnovGenome(int id,
               const std::vector<Trait> &t,
               const std::vector<InnovNodeGene> &n,
               const std::vector<InnovLinkGene> &g);

		//Destructor kills off all lists (including the trait vector)
		~InnovGenome();

		// ******* MUTATORS *******

        //todo: make specific mutators private?

		// ****** MATING METHODS ***** 

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

        Trait &get_trait(const InnovNodeGene &node);
        Trait &get_trait(const InnovLinkGene &gene);

        InnovNodeGene *get_node(int id);

	private:

        static bool linklist_cmp(const InnovLinkGene &a, const InnovLinkGene &b) {
            return a.innovation_num < b.innovation_num;
        }

		//Inserts a InnovNodeGene into a given ordered list of InnovNodeGenes in order
		static void add_node(std::vector<InnovNodeGene> &nlist, const InnovNodeGene &n);

		//Adds a new gene that has been created through a mutation in the
		// *correct order* into the list of links in the genome
		static void add_link(std::vector<InnovLinkGene> &glist, const InnovLinkGene &g);

    private:
        void delete_link(InnovLinkGene *link);

*/
    };
}

