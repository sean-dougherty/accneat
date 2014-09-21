#include "deme.h"

using namespace NEAT;
using namespace std;

//todo: put in env
#define TOURNAMENT_SIZE 5

Deme::Deme(rng_t rng_,
           vector<unique_ptr<Genome>> &seeds_,
           size_t size_,
           size_t population_index_)
    : orgs(rng_, seeds_, size_, population_index_)
    , population_index(population_index_)
    , generation(0) {

    for(Organism &org: orgs.curr()) {
        org.create_phenotype();
    }
}

Deme::~Deme() {
}

void Deme::evaluate(std::function<void (Organism &org)> eval_org) {
#pragma omp parallel for
    for(size_t i = 0; i < orgs.size(); i++) {
        Organism &org = orgs.curr()[i];
        eval_org(org);
    }
}

Organism &Deme::get_fittest() {
    Organism *best = &orgs.curr()[0];
    for(size_t i = 1; i < orgs.size(); i++) {
        Organism *candidate = &orgs.curr()[i];
        if(candidate->fitness > best->fitness) {
            best = candidate;
        }
    }
    return *best;
}

void Deme::next_generation(std::vector<Organism *> &elites,
                           PopulationInnovations &innovations) {
    generation++;
    orgs.next_generation(generation);

    size_t nparents = elites.size() + orgs.size();
    auto get_random_parent = [this, &elites, nparents](rng_t &rng) {
        size_t i = rng.integer(0, nparents - 1);
        return i < elites.size()
        ? elites[i]
        : &orgs.prev()[i - elites.size()];
    };

    auto get_tournament_winner = [&get_random_parent](rng_t &rng) {
        Organism *parent = get_random_parent( rng);
        for(size_t i = 0; i < (TOURNAMENT_SIZE-1); i++) {
            Organism *candidate = get_random_parent(rng);
            if(candidate->fitness > parent->fitness) {
                parent = candidate;
            }
        }
        return parent;
    };

#pragma omp parallel for
    for(size_t i = 0; i < orgs.size(); i++) {
        Organism *offspring = &orgs.curr()[i];
        rng_t &rng = offspring->genome->rng;

        Organism *parent1 = get_tournament_winner(rng);
        Organism *parent2 = get_tournament_winner(rng);

        assert(parent1->generation < offspring->generation);
        assert(parent2->generation < offspring->generation);

        auto create_innov = [offspring, i, &innovations] (InnovationId id,
                                                          InnovationParms parms,
                                                          IndividualInnovation::ApplyFunc apply) {
            innovations.add(IndividualInnovation(offspring->population_index,
                                                 id,
                                                 parms,
                                                 apply));
        };

        Genome::mate(create_innov,
                     parent1->genome,
                     parent2->genome,
                     offspring->genome,
                     parent1->fitness,
                     parent2->fitness);        
    }
}

void Deme::create_phenotypes() {
    //Create the neural nets for the new organisms.
#pragma omp parallel for
    for(size_t iorg = 0; iorg < orgs.size(); iorg++) {
        orgs.curr()[iorg].create_phenotype();
    }
}

void Deme::write(std::ostream& out) {
    for(Organism &org: orgs.curr()) {
        org.genome->print(out);
    }
}

void Deme::verify() {
    for(Organism &org: orgs.curr()) {
        org.genome->verify();
    }
}
