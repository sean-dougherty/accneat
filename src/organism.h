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
		real_t fitness;  //A measure of fitness for the Organism
		real_t orig_fitness;  //A fitness measure that won't change during adjustments
		real_t error;  //Used just for reporting purposes
		bool winner;  //Win marker (if needed for a particular task)
		Network net;  //The Organism's phenotype
		Genome genome; //The Organism's genotype 
		class Species *species;  //The Organism's Species 
		real_t expected_offspring; //Number of children this Organism may have
		int generation;  //Tells which generation this Organism is from
		bool eliminate;  //Marker for destruction of inferior Organisms
		bool champion; //Marks the species champ
		int super_champ_offspring;  //Number of reserved offspring for a population leader
		int time_alive; //When playing in real-time allows knowing the maturity of an individual

		// MetaData for the object
		char metadata[128];
		bool modified;

		// Print the Organism's genome to a file preceded by a comment detailing the organism's species, number, and fitness 
		void write(std::ostream &outFile);

		Organism();
		Organism(const Organism& org) {trap("shouldn't ever have to copy");}
		~Organism();

        void init(real_t fit, int gen, const char *metadata = nullptr);

        void create_phenotype();
	};

	// This is used for list sorting of Organisms by fitness..highest fitness first
	bool order_orgs(Organism *x, Organism *y);

	bool order_orgs_by_adjusted_fit(Organism *x, Organism *y);

} // namespace NEAT

