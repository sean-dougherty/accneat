#include "innovgenomemanager.h"

#include "genome.h"
#include "util.h"

using namespace NEAT;
using namespace std;

InnovGenomeManager::~InnovGenomeManager() {
}

vector<unique_ptr<Genome>> InnovGenomeManager::create_seed_generation(size_t ngenomes,
                                                                      rng_t rng,
                                                                      size_t ntraits,
                                                                      size_t ninputs,
                                                                      size_t noutputs,
                                                                      size_t nhidden) {
    Genome start_genome(rng,
                        ntraits,
                        ninputs,
                        noutputs,
                        nhidden);

    vector<unique_ptr<Genome>> genomes;
    {
        rng_t _rng = rng;
        for(int i = 0; i < NEAT::pop_size; i++) {
            genomes.emplace_back(make_unique<Genome>());
            Genome &g = *genomes.back();
            start_genome.duplicate_into(&g);
            g.rng.seed(_rng.integer());
            g.mutate_link_weights(1.0,1.0,COLDGAUSSIAN);
            g.randomize_traits();
        }
    }

	//Keep a record of the innovation and node number we are on
    innovations.init(genomes.back()->get_last_node_id(),
                     genomes.back()->get_last_gene_innovnum());

    return genomes;
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
