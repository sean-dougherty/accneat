#include "std.h" // Must be included first. Precompiled header with standard library includes.
#include "innovgenome.h"
#include "innovgenomemanager.h"
#include "util.h"

using namespace NEAT;
using namespace std;

/* Colin's values
#define MAX_COMPLEXIFY_PHASE_DURATION 100
#define MAX_PRUNE_PHASE_DURATION 0.3
*/
#define MAX_COMPLEXIFY_PHASE_DURATION 40
#define PRUNE_PHASE_FACTOR 0.5

InnovGenomeManager::InnovGenomeManager() {
    if(env->search_type == GeneticSearchType::PHASED) {
        search_phase = COMPLEXIFY;
        search_phase_start = 1;
        max_phase_duration = MAX_COMPLEXIFY_PHASE_DURATION;
    } else {
        search_phase = UNDEFINED;
        search_phase_start = -1;
        max_phase_duration = 0;
    }
    generation = 1;
}

InnovGenomeManager::~InnovGenomeManager() {
}

static InnovGenome *to_innov(Genome &g) {
    return dynamic_cast<InnovGenome *>(&g);
}

unique_ptr<Genome> InnovGenomeManager::make_default() {
    return unique_ptr<Genome>(new InnovGenome());
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
        for(int i = 0; i < env->pop_size; i++) {
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
    return to_innov(genome1)->compatibility(to_innov(genome2)) < env->compat_threshold;
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

    if(!is_mate_allowed()) {
        if(fitness1 > fitness2) {
            clone(genome1, offspring);
        } else {
            clone(genome2, offspring);
        }
        mutate(offspring, MUTATE_OP_ANY);
    } else {
        InnovGenome::mate(to_innov(genome1),
                          to_innov(genome2),
                          to_innov(offspring),
                          fitness1,
                          fitness2);

        //Determine whether to mutate the baby's InnovGenome
        //This is done randomly or if the genome1 and genome2 are the same organism
        if( !offspring.rng.under(env->mate_only_prob) ||
            (genome2.genome_id == genome1.genome_id) ||
            (to_innov(genome2)->compatibility(to_innov(genome1)) == 0.0) ) {

            mutate(offspring, MUTATE_OP_ANY);
        }
    }
}

void InnovGenomeManager::mutate(Genome &genome_,
                                MutationOperation op) {
    InnovGenome *genome = to_innov(genome_);
    bool allow_del = is_delete_allowed();
    bool allow_add = is_add_allowed();

    switch(op) {
    case MUTATE_OP_WEIGHTS:
        genome->mutate_link_weights(env->weight_mut_power,
                                    1.0,
                                    GAUSSIAN);
        break;
    case MUTATE_OP_STRUCTURE: {
        if(!allow_add && !allow_del) {
            mutate(genome_, MUTATE_OP_WEIGHTS);
        } else {
            if(!allow_del || genome_.rng.boolean()) {
                genome->mutate_add_link(create_innov_func(genome_),
                                        env->newlink_tries);
            } else {
                genome->mutate_delete_link();
            }
        }
    } break;
    case MUTATE_OP_ANY: {
        rng_t &rng = genome->rng;
        rng_t::prob_switch_t op = rng.prob_switch();

        if( allow_add && op.prob_case(env->mutate_add_node_prob) ) {
            bool delete_split_link = env->search_type != GeneticSearchType::COMPLEXIFY;
            genome->mutate_add_node(create_innov_func(genome_), delete_split_link);
        } else if( allow_add && op.prob_case(env->mutate_add_link_prob) ) {
            genome->mutate_add_link(create_innov_func(genome_),
                                    env->newlink_tries);
        } else if( allow_del && op.prob_case(env->mutate_delete_link_prob) ) {
            genome->mutate_delete_link();
        } else if( allow_del && op.prob_case(env->mutate_delete_node_prob) ) {
            genome->mutate_delete_node();
        } else {
            //Only do other mutations when not doing sturctural mutations
            if( rng.under(env->mutate_random_trait_prob) ) {
                genome->mutate_random_trait();
            }
            if( rng.under(env->mutate_link_trait_prob) ) {
                genome->mutate_link_trait(1);
            }
            if( rng.under(env->mutate_node_trait_prob) ) {
                genome->mutate_node_trait(1);
            }
            if( rng.under(env->mutate_link_weights_prob) ) {
                genome->mutate_link_weights(env->weight_mut_power,
                                            1.0,
                                            GAUSSIAN);
            }

            if(env->search_type == GeneticSearchType::COMPLEXIFY) {
                if( rng.under(env->mutate_toggle_enable_prob) ) {
                    genome->mutate_toggle_enable(1);
                }
                if (rng.under(env->mutate_gene_reenable_prob) ) {
                    genome->mutate_gene_reenable(); 
                }
            }
        }
    } break;
    default:
        panic();
    }

    if(genome->links.size() == 0) {
        genome->mutate_add_link(create_innov_func(genome_),
                                env->newlink_tries);
    }
}

void InnovGenomeManager::finalize_generation(bool new_fittest) {
    innovations.apply();

    generation++;
    if(env->search_type == GeneticSearchType::PHASED) {
        int phase_duration = generation - search_phase_start;
        switch(search_phase) {
        case COMPLEXIFY:
            if( (phase_duration >= max_phase_duration) 
                || new_fittest) {
                cout << "phase PRUNE @ gen " << generation << endl;
                search_phase_start = generation;
                search_phase = PRUNE;
                max_phase_duration = 1 + int(PRUNE_PHASE_FACTOR * phase_duration);
            }
            break;
        case PRUNE:
            if(phase_duration >= max_phase_duration) {
                cout << "phase COMPLEXIFY @ gen " << generation << endl;
                search_phase_start = generation;
                search_phase = COMPLEXIFY;
                max_phase_duration = MAX_COMPLEXIFY_PHASE_DURATION;
            }
            break;
        default:
            panic();
        }
    }
}

CreateInnovationFunc InnovGenomeManager::create_innov_func(Genome &g) {
    return [this, &g] (InnovationId id,
                       InnovationParms parms,
                       IndividualInnovation::ApplyFunc apply) {
        innovations.add(IndividualInnovation(g.genome_id, id, parms, apply));
    };
}

bool InnovGenomeManager::is_mate_allowed() {
    switch(env->search_type) {
    case GeneticSearchType::PHASED:
        return search_phase == COMPLEXIFY;
    case GeneticSearchType::BLENDED:
    case GeneticSearchType::COMPLEXIFY:
        return true;
    default:
        panic();
    }
}

bool InnovGenomeManager::is_add_allowed() {
    switch(env->search_type) {
    case GeneticSearchType::PHASED:
        return search_phase == COMPLEXIFY;
    case GeneticSearchType::BLENDED:
    case GeneticSearchType::COMPLEXIFY:
        return true;
    default:
        panic();
    }
}

bool InnovGenomeManager::is_delete_allowed() {
    switch(env->search_type) {
    case GeneticSearchType::PHASED:
        return search_phase == PRUNE;
    case GeneticSearchType::BLENDED:
        return true;
    case GeneticSearchType::COMPLEXIFY:
        return false;
    default:
        panic();
    }
}
