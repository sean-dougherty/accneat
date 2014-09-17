#pragma once

#include "population.h"

#include "innovation.h"
#include "organismsbuffer.h"

namespace NEAT {

    class DemesPopulation : public Population {
    public:
		DemesPopulation(rng_t &rng, Genome *g, int size);
		virtual ~DemesPopulation();

        virtual bool evaluate(std::function<void (Organism &org)> eval_org) override;
        virtual class Organism &get_fittest() override {return fittest;}
		virtual void next_generation() override;
		virtual void verify() override;

		virtual void write(std::ostream& out) override;

    private:
        int generation;
        OrganismsBuffer orgs;
        Organism fittest;
        PopulationInnovations innovations;
    };

}
