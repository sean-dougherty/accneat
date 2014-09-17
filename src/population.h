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
#ifndef _POPULATION_H_
#define _POPULATION_H_

#include <cmath>
#include <vector>
#include "innovation.h"
#include "genome.h"
#include "species.h"
#include "organism.h"
#include "organismsbuffer.h"

#include <assert.h>

namespace NEAT {

	// ---------------------------------------------  
	// POPULATION CLASS:
	//   A Population is a group of Organisms   
	//   including their species                        
	// ---------------------------------------------  
	class Population {
	public:
		// Construct off of a single spawning Genome 
		Population(rng_t &rng, Genome *g, int size);
		~Population();

		// Turnover the population to a new generation using fitness 
		// The generation argument is the next generation
		bool epoch(int generation);

        size_t size() {return orgs.size();}
        Organism *get(size_t i) {return &orgs.curr()[i];}

		// Write Population to a stream (e.g. file) in speciated order with comments separating each species
		void write(std::ostream& out);

		// Run verify on all Genomes in this Population (Debugging)
		void verify();

    private:
		bool spawn(Genome *g);
		bool speciate();

        OrganismsBuffer orgs;

        std::vector<Species*> species;  // Species in the Population. Note that the species should comprise all the genomes 
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

#endif
