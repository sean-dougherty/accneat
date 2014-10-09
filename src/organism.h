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
#include "network.h"

namespace NEAT {

    struct OrganismEvaluation {
        real_t fitness = 0.0;
        real_t error = 0.0;
        
        void reset() {fitness = error = 0.0;}
    };

	// ---------------------------------------------  
	// ORGANISM CLASS:
	//   Organisms are Genomes and Networks with fitness
	//   information 
	//   i.e. The genotype and phenotype together
	// ---------------------------------------------  
	class Organism {
	public:
        size_t population_index; //Unique within the population,always in [0, env->pop_size).
                                 //Provides client with convenient storage of associated
                                 //data in an array.

        OrganismEvaluation eval;
		Network net;  //The Organism's phenotype
        std::unique_ptr<Genome> genome; //The Organism's genotype
		int generation;  //Tells which generation this Organism is from

        Organism(const Organism &other);
        Organism(const Genome &genome);
		virtual ~Organism();

        virtual void init(int gen);

        Organism &operator=(const Organism &other);

        virtual void write(std::ostream &out) const;

    protected:
        Organism() {}
        virtual void copy_into(Organism &dst) const;
	};

} // namespace NEAT

