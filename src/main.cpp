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
#include <vector>
#include "neat.h"
#include "population.h"
#include "seq_experiment.h"
#include "timer.h"
using namespace std;

#if false
#define RAND_SEED (unsigned)time( NULL )
#else
#define RAND_SEED 0
#endif



int main(int argc, char *argv[]) {
    NEAT::Population *p = nullptr;

    /* Seed the random-number generator with current time so that
       the numbers will be different every time we run.    */
    srand( RAND_SEED );

    if (argc != 4) {
        cerr << "usage: maxgens neat ne_path startgenes_path" << endl;
        return -1;
    }

    int maxgens = atoi(argv[1]);

    //Load in the params
    NEAT::load_neat_params(argv[2],true);

    p = seq_experiment(maxgens, argv[3]);

    if (p)
        delete p;

    Timer::report();

    return(0);
 
}

