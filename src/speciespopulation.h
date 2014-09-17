/*
 Copyright 2001 The University of Texas at Austin

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/
#pragma once 

#include "innovation.h"
#include "organismsbuffer.h"
#include "population.h"
#include <vector>

namespace NEAT {

	class SpeciesPopulation : public Population {
	public:
		// Construct off of a single spawning Genome 
		SpeciesPopulation(rng_t &rng, Genome *g, int size);
		virtual ~SpeciesPopulation();

		virtual void next_generation() override;

        virtual size_t size() override {return orgs.size();}
        virtual Organism *get(size_t i) override {return &orgs.curr()[i];}

		// Write SpeciesPopulation to a stream (e.g. file) in speciated order with comments separating each species
		virtual void write(std::ostream& out) override;

		// Run verify on all Genomes in this SpeciesPopulation (Debugging)
		virtual void verify() override;

    private:
		bool spawn(Genome *g);
		bool speciate();

        int generation;
        OrganismsBuffer orgs;

        std::vector<class Species*> species;  // Species in the SpeciesPopulation. Note that the species should comprise all the genomes 
        PopulationInnovations innovations;

		// ******* Member variables used during reproduction *******
		int last_species;  //The highest species number

		// ******* Fitness Statistics *******
		real_t mean_fitness;
		real_t variance;
		real_t standard_deviation;

		int winnergen; //An integer that when above zero tells when the first winner appeared

		// ******* When do we need to delta code? *******
		real_t highest_fitness;  //Stagnation detector
		int highest_last_changed; //If too high, leads to delta coding
	};

} // namespace NEAT
