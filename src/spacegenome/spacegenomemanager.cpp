#include "spacegenomemanager.h"

#include "spacegenome.h"
#include "util.h"

using namespace NEAT;
using namespace std;

SpaceGenomeManager::~SpaceGenomeManager() {
}

static SpaceGenome *to_space(Genome &g) {
    return dynamic_cast<SpaceGenome *>(&g);
}

vector<unique_ptr<Genome>> SpaceGenomeManager::create_seed_generation(size_t ngenomes,
                                                                      rng_t rng,
                                                                      size_t ntraits,
                                                                      size_t ninputs,
                                                                      size_t noutputs,
                                                                      size_t nhidden) {
    trap("need to implement support for search_type");

    SpaceGenome start_genome(rng,
                             ntraits,
                             ninputs,
                             noutputs,
                             nhidden);

    vector<unique_ptr<Genome>> genomes;
    {
        rng_t _rng = rng;
        for(int i = 0; i < env->pop_size; i++) {
            SpaceGenome *g = new SpaceGenome();
            start_genome.duplicate_into(g);
            g->rng.seed(_rng.integer());
            g->mutate_link_weights(1.0,1.0,COLDGAUSSIAN);
            g->randomize_traits();
            
            genomes.emplace_back(unique_ptr<Genome>(g));
        }
    }

    return genomes;
}

//todo: implement
bool SpaceGenomeManager::are_compatible(Genome &genome1,
                                        Genome &genome2) {
    return true;
}

void SpaceGenomeManager::clone(Genome &orig,
                               Genome &clone) {
    to_space(orig)->duplicate_into(to_space(clone));
}

void SpaceGenomeManager::mate(Genome &genome1,
                              Genome &genome2,
                              Genome &offspring,
                              real_t fitness1,
                              real_t fitness2) {

    SpaceGenome::mate(to_space(genome1),
                      to_space(genome2),
                      to_space(offspring),
                      fitness1,
                      fitness2);        
}
 

void SpaceGenomeManager::mutate(Genome &genome,
                                MutationOperation op) {
    switch(op) {
    case MUTATE_OP_WEIGHTS:
        to_space(genome)->mutate_link_weights(env->weight_mut_power,
                                              1.0,
                                              GAUSSIAN);
        break;
    case MUTATE_OP_STRUCTURE:
        //todo: other operations as well?
        to_space(genome)->mutate_add_link();
        break;
    case MUTATE_OP_ANY:
        to_space(genome)->mutate();
        break;
    default:
        panic();
    }
}

void SpaceGenomeManager::finalize_generation() {
    //no-op
}

