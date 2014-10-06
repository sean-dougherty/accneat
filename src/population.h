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
#include "genomemanager.h"
#include "rng.h"

namespace NEAT {

    class Population {
    public:
        static Population *create(rng_t rng,
                                  GenomeManager *genome_manager,
                                  std::vector<std::unique_ptr<Genome>> &seeds);

        virtual ~Population() {}

        //Executes eval_org for each Organism in the population.
        //Returns true if a new fittest was found.
        virtual bool evaluate(std::function<void (class Organism &org)> eval_org) = 0;

        virtual class Organism &get_fittest() = 0;
		virtual void next_generation() = 0;
		virtual void verify() = 0;

		virtual void write(std::ostream& out) = 0;
    };

} // namespace NEAT

