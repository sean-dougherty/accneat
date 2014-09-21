#pragma once

#include "deme.h"
#include "population.h"
#include "innovation.h"

namespace NEAT {

    class DemesPopulation : public Population {
    public:
		DemesPopulation(rng_t rng, std::vector<std::unique_ptr<Genome>> &seeds);
		virtual ~DemesPopulation();

        virtual bool evaluate(std::function<void (Organism &org)> eval_org) override;
        virtual class Organism &get_fittest() override;
		virtual void next_generation() override;
		virtual void verify() override;

		virtual void write(std::ostream& out) override;

    private:
        PopulationInnovations innovations;
        int generation;
        std::vector<Deme> demes;
        std::vector<Organism *> global_elites;
        std::vector<Organism *> elites;
    };

}
