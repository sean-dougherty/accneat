#include "experiment.h"
#include "network.h"
#include "population.h"
#include "timer.h"
#include <assert.h>
#include <fstream>

using namespace NEAT;
using namespace std;

#define NACTIVATES_PER_INPUT 10

real_t Step::err(Network *net,
                 float **details_act,
                 float **details_err) {
    real_t result = 0.0;

    for(size_t i = 0; i < output.size(); i++) {
        real_t err = abs(net->get_output(i) - output[i]);

        if(err < 0.05) {
            err = 0.0;
        }

        result += err;

        **details_act = net->get_output(i);
        **details_err = err;

        (*details_act)++;
        (*details_err)++;
    }

    return result * weight;
}

map<string, Experiment *> *Experiment::experiments = nullptr;

Experiment *Experiment::get(const char *name) {
    if(experiments == nullptr) {
        experiments = new map<string, Experiment*>();
    }
    return experiments->find(name)->second;
}

Experiment::Experiment(const char *name) {
    if(get(name) != nullptr) {
        trap("Experiment already registered: " << name);
    }
    experiments->insert(make_pair(name, this));
}

real_t Experiment::score(real_t errorsum) {
    real_t x = 1.0 - errorsum/max_err;
    return x * x;
};

void Experiment::print(Population *pop, int gen) {
    char filename[1024];
    sprintf(filename, "gen_%d", gen);
    ofstream out(filename);
    pop->write(out);
}

void Experiment::init() {
    init_env();
    tests = create_tests();

    // Validate tests
    {
        assert(tests.size() > 0);
        assert(tests[0].steps.size() > 0);

        size_t ninputs = tests[0].steps[0].input.size();
        assert(ninputs > 0);

        size_t noutputs = tests[0].steps[0].output.size();
        assert(noutputs > 0);

        for(size_t i = 1; i < tests.size(); i++) {
            Test &test = tests[i];
            assert(test.steps.size() > 0);
            for(Step &step: tests[i].steps) {
                assert(step.input.size() == ninputs);
                assert(step.output.size() == noutputs);
            }
        }
    }

    // Determine total number of steps
    {
        size_t n = 0;
        for(auto &test: tests) {
            n += test.steps.size();
        }
        nsteps = n;
    }

    // Determine total number of outputs
    nouts = nsteps * tests[0].steps[0].output.size();

    // Determine maximum error value
    {
        real_t total = 0.0;
        for(auto &t: tests) {
            for(auto &s: t.steps) {
                total += s.weight * s.output.size();
            }
        }
        max_err = total;
    }

    details_act = new float[NEAT::pop_size * nouts];
    details_err = new float[NEAT::pop_size * nouts];
}

void Experiment::run(rng_t &rng, int gens) {
    GenomeManager *genome_manager = GenomeManager::create();
    vector<unique_ptr<Genome>> genomes = 
        genome_manager->create_seed_generation(NEAT::pop_size,
                                               rng,
                                               1,
                                               tests[0].steps[0].input.size(),
                                               tests[0].steps[0].output.size(),
                                               3);
            

    int nsuccesses = 0;

    for(int expcount = 0; expcount < NEAT::num_runs; expcount++) {
        //Spawn the Population
        Population *pop = Population::create(rng, genome_manager, genomes);
      
        bool success = false;
        int gen;
        for(gen = 1; !success && (gen <= gens); gen++) {
            cout << "Epoch " << gen << endl;	

            static Timer timer("epoch");
            timer.start();

            if(gen != 1) {
                pop->next_generation();
            }

            evaluate(pop);

            if(is_success(&pop->get_fittest())) {
                success = true;
                nsuccesses++;
            }

            timer.stop();
            Timer::report();

            //Don't print on success because we'll exit the loop and print then.
            if(!success && (gen % NEAT::print_every == 0))
                print(pop, gen);
        }

        print(pop, gen - 1);

        delete pop;
    }

    cout << "Failures: " << (NEAT::num_runs - nsuccesses) << " out of " << NEAT::num_runs << " runs" << endl;
}

bool Experiment::is_success(Organism *org) {
    return org->error <= 0.0000001;
}

void Experiment::evaluate(Population *pop) {
    bool new_fittest = pop->evaluate([this](Organism &org) {evaluate_org(org);});
    Organism &fittest = pop->get_fittest();

    if(new_fittest) {
        float *fittest_act = details_act + fittest.population_index * nouts;
        float *fittest_err = details_err + fittest.population_index * nouts;
        
        for(auto &test: tests) {
            for(auto &step: test.steps) {
                for(size_t i = 0; i < step.output.size(); i++) {
                    printf("%f (%f) ", *fittest_act++, *fittest_err++);
                }
                printf("\n");
            }
            printf("---\n");
        }
    }

    Genome::Stats gstats = fittest.genome->get_stats();
    cout << "fittest [" << fittest.population_index << "]: fitness=" << fittest.fitness << ", error=" << fittest.error << ", nnodes=" << gstats.nnodes << ", nlinks=" << gstats.nlinks << endl;
}

void Experiment::evaluate_org(Organism &org) {
    float *details_act = this->details_act + org.population_index * nouts;
    float *details_err = this->details_err + org.population_index * nouts;
    Network *net = &org.net;
    real_t errorsum = 0.0;

    for(auto &test: tests) {
        for(auto &step: test.steps) {
            net->load_sensors(step.input);
            for(size_t i = 0; i < NACTIVATES_PER_INPUT; i++) {
                net->activate();
            }
            errorsum += step.err( net, &details_act, &details_err );
        }

        net->flush();
    }
 
    org.fitness = score(errorsum);
    org.error = errorsum;
}
