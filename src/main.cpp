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
#include <iostream>
#include <string>
#include "neat.h"
#include "experiment.h"
#include "util.h"
using namespace std;


int main(int argc, char *argv[]) {

    if (argc != 7) {
        cerr << "usage: neat experiment_name rng_seed pop_type genome_type pop_size maxgens" << endl;
        return -1;
    }

    int argi = 1;
    const char *experiment_name = argv[argi++];
    int rng_seed = stoi(argv[argi++]);
    string pop_type = argv[argi++];
    string genome_type = argv[argi++];
    int pop_size = stoi(argv[argi++]);
    int maxgens = stoi(argv[argi++]);

    if(pop_type == "s") {
        NEAT::population_type = NEAT::PopulationType::SPECIES;
    } else if(pop_type == "d") {
        NEAT::population_type = NEAT::PopulationType::DEMES;
    } else {
        panic();
    }

    if(genome_type == "i") {
        NEAT::genome_type = NEAT::GenomeType::INNOV;
    } else if(genome_type == "s") {
        NEAT::genome_type = NEAT::GenomeType::SPACE;
    } else {
        panic();
    }

    NEAT::rng_t rng;
    rng.seed(rng_seed);

    NEAT::pop_size = pop_size;

    NEAT::Experiment *exp = NEAT::Experiment::get(experiment_name);
    if(exp == nullptr) {
        trap("No such experiment: " << experiment_name);
    }
    exp->init();
    exp->run(rng, maxgens);

    return(0);
}

