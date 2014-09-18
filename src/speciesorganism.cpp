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
	expected_offspring=0;
	eliminate=false;
	champion=false;
	super_champ_offspring=0;
}
