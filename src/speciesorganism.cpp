#include "speciesorganism.h"

#include "species.h"

using namespace NEAT;

SpeciesOrganism::SpeciesOrganism()
    : Organism() {
}

SpeciesOrganism::~SpeciesOrganism() {
}

void SpeciesOrganism::init(int gen) {
    Organism::init(gen);

	species = nullptr;  //Start it in no Species
	adjusted_fitness=0.0;
	expected_offspring=0;
	eliminate=false;
	champion=false;
	super_champ_offspring=0;
}

void SpeciesOrganism::copy_into(Organism &dst_) const {
    Organism::copy_into(dst_);

    SpeciesOrganism *dst = dynamic_cast<SpeciesOrganism *>(&dst_);

#define copy(field) dst->field = this->field;
    
    copy(species);
    copy(adjusted_fitness);
    copy(expected_offspring);
    copy(eliminate);
    copy(champion);
    copy(super_champ_offspring);

#undef copy
}
