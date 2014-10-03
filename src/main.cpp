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
#include <unistd.h>
#include "neat.h"
#include "experiment.h"
#include "util.h"
using namespace std;

#define DEFAULT_RNG_SEED 1
#define DEFAULT_MAX_GENS 10000

void usage() {
    cerr << "usage: neat [OPTIONS]... experiment_name" << endl;
    cerr << endl;
    cerr << "experiment names: ";
    auto names = NEAT::Experiment::get_names();
    for(size_t i = 0; i < names.size(); i++) {
        if(i != 0)
            cerr << ", ";
        cerr << names[i];
    }
    cerr << endl;
    cerr << endl;

    cerr << "OPTIONS" << endl;
    cerr << "  -c num_experiments (default=" << NEAT::num_runs << ")" << endl;
    cerr << "  -r RNG_seed (default=" << DEFAULT_RNG_SEED << ")" << endl;
    cerr << "  -n population_size (default=" << NEAT::pop_size << ")" << endl;
    cerr << "  -x max_generations (default=" << DEFAULT_MAX_GENS << ")" << endl;
    cerr << endl;
    cerr << "ILL-ADVISED OPTIONS" << endl;
    cerr << "  -p population_type {species, demes} (default=species)" << endl;
    cerr << "  -g genome_type {innov, space} (default=innov)" << endl;

    exit(1);
}

template<typename T>
T parse_enum(const char *opt, string str, map<string,T> vals) {
    auto it = vals.find(str);
    if(it == vals.end()) {
        error("Invalid value for " << opt << ": " << str);
    }
    return it->second;
}

int parse_int(const char *opt, const char *str) {
    try {
        return stoi(str);
    } catch(...) {
        error("Expecting integer argument for " << opt << ", found '" << str << "'.");
    }
}

int main(int argc, char *argv[]) {

    int rng_seed = DEFAULT_RNG_SEED;
    int maxgens = DEFAULT_MAX_GENS;
    {
        int opt;
        while( (opt = getopt(argc, argv, "c:r:p:g:n:x:")) != -1) {
            switch(opt) {
            case 'c':
                NEAT::num_runs = parse_int("-c", optarg);
                break;
            case 'r':
                rng_seed = parse_int("-r", optarg);
                break;
            case 'p':
                NEAT::population_type = parse_enum<NEAT::PopulationType>("-p", optarg, {
                        {"species", NEAT::PopulationType::SPECIES},
                        {"demes", NEAT::PopulationType::DEMES}
                    });
                break;
            case 'g':
                NEAT::genome_type = parse_enum<NEAT::GenomeType>("-g", optarg, {
                        {"innov", NEAT::GenomeType::INNOV},
                        {"space", NEAT::GenomeType::SPACE}
                    });
                break;
            case 'n':
                NEAT::pop_size = parse_int("-n", optarg);
                break;
            case 'x':
                maxgens = parse_int("-x", optarg);
                break;
            default:
                error("Invalid option: -" << (char)opt);
            }
        }
    }

    int nargs = argc - optind;
    if(nargs == 0) {
        usage();
    } else if(nargs > 1) {
        error("Unexpected argument: " << argv[optind+1]);
    }

    const char *experiment_name = argv[optind++];

    NEAT::Experiment *exp = NEAT::Experiment::get(experiment_name);
    if(exp == nullptr) {
        trap("No such experiment: " << experiment_name);
    }

    NEAT::rng_t rng{rng_seed};
    exp->init();
    exp->run(rng, maxgens);

    return(0);
}

