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
#include "seq_experiment.h"
#include "util.h"
using namespace std;


int main(int argc, char *argv[]) {

    if (argc != 5) {
        cerr << "usage: neat rng_seed pop_type pop_size maxgens" << endl;
        return -1;
    }

    int argi = 1;
    int rng_seed = stoi(argv[argi++]);
    string pop_type = argv[argi++];
    int pop_size = stoi(argv[argi++]);
    int maxgens = stoi(argv[argi++]);

    if(pop_type == "s") {
        NEAT::population_type = NEAT::PopulationType::SPECIES;
    } else if(pop_type == "d") {
        NEAT::population_type = NEAT::PopulationType::DEMES;
    } else {
        panic();
    }

    NEAT::rng_t rng;
    rng.seed(rng_seed);

    NEAT::pop_size = pop_size;

    seq_experiment(rng, maxgens);

    return(0);
}

