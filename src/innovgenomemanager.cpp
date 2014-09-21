#include "innovgenomemanager.h"

#include "genome.h"
#include "util.h"

using namespace NEAT;
using namespace std;

InnovGenomeManager::~InnovGenomeManager() {
}

static InnovGenome *to_innov(Genome &g) {
    return dynamic_cast<InnovGenome *>(&g);
}

vector<unique_ptr<Genome>> InnovGenomeManager::create_seed_generation(size_t ngenomes,
                                                                      rng_t rng,
                                                                      size_t ntraits,
                                                                      size_t ninputs,
                                                                      size_t noutputs,
                                                                      size_t nhidden) {
    InnovGenome start_genome(rng,
                             ntraits,
                             ninputs,
                             noutputs,
                             nhidden);

    vector<unique_ptr<Genome>> genomes;
    {
        rng_t _rng = rng;
        for(int i = 0; i < NEAT::pop_size; i++) {
            InnovGenome *g = new InnovGenome();
            start_genome.duplicate_into(g);
            g->rng.seed(_rng.integer());
            g->mutate_link_weights(1.0,1.0,COLDGAUSSIAN);
            g->randomize_traits();
            
            genomes.emplace_back(unique_ptr<Genome>(g));
        }
    }

    {
        InnovGenome *g = to_innov(*genomes.back());

        //Keep a record of the innovation and node number we are on
        innovations.init(g->get_last_node_id(),
                         g->get_last_gene_innovnum());
    }

    return genomes;
}

bool InnovGenomeManager::are_compatible(Genome &genome1,
                                        Genome &genome2) {
    return to_innov(genome1)->compatibility(to_innov(genome2)) < NEAT::compat_threshold;
}

void InnovGenomeManager::clone(Genome &orig,
                               Genome &clone) {
    to_innov(orig)->duplicate_into(to_innov(clone));
}

void InnovGenomeManager::mate(Genome &genome1,
                              Genome &genome2,
                              Genome &offspring,
                              real_t fitness1,
                              real_t fitness2) {

    InnovGenome::mate(create_innov_func(offspring),
                      to_innov(genome1),
                      to_innov(genome2),
                      to_innov(offspring),
                      fitness1,
                      fitness2);        
}
 

void InnovGenomeManager::mutate(Genome &genome,
                                MutationOperation op) {
    switch(op) {
    case MUTATE_OP_WEIGHTS:
        to_innov(genome)->mutate_link_weights(NEAT::weight_mut_power,
                                              1.0,
                                              GAUSSIAN);
        break;
    case MUTATE_OP_STRUCTURE:
        //todo: other operations as well?
        to_innov(genome)->mutate_add_link(create_innov_func(genome),
                                          NEAT::newlink_tries);
        break;
    case MUTATE_OP_ANY:
        to_innov(genome)->mutate(create_innov_func(genome));
        break;
    default:
        panic();
    }
}

void InnovGenomeManager::finalize_generation() {
    innovations.apply();
}

CreateInnovationFunc InnovGenomeManager::create_innov_func(Genome &g) {
    return [this, &g] (InnovationId id,
                       InnovationParms parms,
                       IndividualInnovation::ApplyFunc apply) {
        innovations.add(IndividualInnovation(g.genome_id, id, parms, apply));
    };
}
