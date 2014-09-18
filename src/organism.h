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

#include "genome.h"
#include "util.h"

#include <iostream>

namespace NEAT {

	class Species;

	// ---------------------------------------------  
	// ORGANISM CLASS:
	//   Organisms are Genomes and Networks with fitness
	//   information 
	//   i.e. The genotype and phenotype together
	// ---------------------------------------------  
	class Organism {
	public:
        size_t population_index; //Unique within the population,always in [0, NEAT::pop_size).
                                 //Provides client with convenient storage of associated
                                 //data in an array.

		real_t fitness;  //A measure of fitness for the Organism
		real_t error;  //Used just for reporting purposes
		bool winner;  //Win marker (if needed for a particular task)
		Network net;  //The Organism's phenotype
		Genome genome; //The Organism's genotype 
		int generation;  //Tells which generation this Organism is from

		real_t orig_fitness;  //A fitness measure that won't change during adjustments
		class Species *species;  //The Organism's Species 
		real_t expected_offspring; //Number of children this Organism may have
		bool eliminate;  //Marker for destruction of inferior Organisms
		bool champion; //Marks the species champ
		int super_champ_offspring;  //Number of reserved offspring for a population leader

		Organism();
		~Organism();

        void init(int gen);

        void create_phenotype();
	};

	// This is used for list sorting of Organisms by fitness..highest fitness first
	bool order_orgs(Organism *x, Organism *y);

	bool order_orgs_by_adjusted_fit(Organism *x, Organism *y);

} // namespace NEAT

