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
#include "population.h"
#include "demespopulation.h"
#include "speciespopulation.h"

using namespace NEAT;
using namespace std;


Population *Population::create(rng_t rng,
                               GenomeManager *genome_manager,
                               vector<unique_ptr<Genome>> &seeds) {

    switch(NEAT::population_type) {
    case PopulationType::SPECIES:
        return new SpeciesPopulation(rng, genome_manager, seeds);
    case PopulationType::DEMES:
        return new DemesPopulation(rng, genome_manager, seeds);
    default:
        trap("invalid pop type");
    }
}
