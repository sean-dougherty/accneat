#pragma once

#include "population.h"

#include "innovation.h"
#include "organismsbuffer.h"

namespace NEAT {

    class DemesPopulation : public Population {
    public:
		DemesPopulation(rng_t &rng, Genome *g, int size);
		virtual ~DemesPopulation();

		virtual void next_generation() override;

        virtual size_t size() override {return orgs.size();}
        virtual Organism *get(size_t i) override {return &orgs.curr()[i];}

		// Write SpeciesPopulation to a stream (e.g. file) in speciated order with comments separating each species
		virtual void write(std::ostream& out) override;

		// Run verify on all Genomes in this SpeciesPopulation (Debugging)
		virtual void verify() override;

    private:
        int generation;
        OrganismsBuffer orgs;
        PopulationInnovations innovations;
    };

}
