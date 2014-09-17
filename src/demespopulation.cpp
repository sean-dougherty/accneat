#include "demespopulation.h"

using namespace NEAT;
using namespace std;

DemesPopulation::DemesPopulation(rng_t &rng, Genome *g,int size)
    : generation(0)
    , orgs(rng, size) {
}

DemesPopulation::~DemesPopulation() {
}

void DemesPopulation::next_generation() {
#ifndef NDEBUG
    for(Organism &org: orgs.curr()) {
        assert(org.generation == generation);
    }
#endif

    generation++;

    trap("implement");
}

void DemesPopulation::write(std::ostream& out) {
    trap("implement");
}

void DemesPopulation::verify() {
    for(auto &org: orgs.curr())
        org.genome.verify();
} 



