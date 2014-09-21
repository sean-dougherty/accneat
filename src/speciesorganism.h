#pragma once

#include "organism.h"

namespace NEAT {

    class SpeciesOrganism : public Organism {
    public:
		class Species *species;  //The Organism's Species 
        real_t adjusted_fitness;
		real_t expected_offspring; //Number of children this Organism may have
		bool eliminate;  //Marker for destruction of inferior Organisms
		bool champion; //Marks the species champ
		int super_champ_offspring;  //Number of reserved offspring for a population leader

        SpeciesOrganism();
		SpeciesOrganism(const SpeciesOrganism &other);
		SpeciesOrganism(const Genome &genome);
        virtual ~SpeciesOrganism();

        virtual void init(int gen) override;

    protected:
        virtual void copy_into(Organism &dst) const override;
    };

}
