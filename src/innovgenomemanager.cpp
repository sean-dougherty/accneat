#include "innovgenomemanager.h"

#include "genome.h"
#include "util.h"

using namespace NEAT;
using namespace std;

InnovGenomeManager::~InnovGenomeManager() {
}

vector<Genome *> InnovGenomeManager::create_seed_generation(size_t ngenomes,
                                                            class rng_t &rng,
                                                            size_t ntraits,
                                                            size_t ninputs,
                                                            size_t noutputs,
                                                            size_t nhidden) {
    vector<Genome *> result;

    Genome *start_genome = Genome::create_seed_genome(rng,
                                                      ntraits,
                                                      ninputs,
                                                      noutputs,
                                                      nhidden);

    for(size_t i = 0; i < ngenomes; i++) {
        Genome *g = new Genome();
        start_genome->duplicate_into(g);
		g->mutate_link_weights(1.0,1.0,COLDGAUSSIAN);
		g->randomize_traits();
	}

	//Keep a record of the innovation and node number we are on
    innovations.init(result.back()->get_last_node_id(),
                     result.back()->get_last_gene_innovnum());
    
    return result;
}

void InnovGenomeManager::mate(Genome *genome1,
                              Genome *genome2,
                              Genome *offspring,
                              real_t fitness1,
                              real_t fitness2) {
    impl();
}
 

void InnovGenomeManager::mutate(Genome *genome) {
    impl();
}
