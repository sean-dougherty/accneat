#include "experiment.h"
#include "network.h"
#include "population.h"
#include "timer.h"
#include <assert.h>
#include <fstream>

using namespace NEAT;
using namespace std;

#define NACTIVATES_PER_INPUT 10

//------------------------------
//---
//--- UTIL
//---
//------------------------------
static string get_dir_path(int experiment_num) {
    char buf[1024];
    sprintf(buf, "./experiment_%d", experiment_num);
    return buf;
}

static string get_fittest_path(int experiment_num, int generation) {
    char buf[1024];
    sprintf(buf, "%s/fittest_%d",
            get_dir_path(experiment_num).c_str(),
            generation);
    return buf;
}

const char *str(Test::Type t) {
    switch(t) {
    case Test::Training: return "Training";
    case Test::Fittest: return "Fittest";
    default: panic();
    }
}

//------------------------------
//---
//--- CLASS Step
//---
//------------------------------
Step::Step(const std::vector<real_t> &input_,
           const std::vector<real_t> &output_,
           real_t weight_)
    : input(input_)
    , output(output_)
    , weight(weight_) {
}

real_t Step::err(Network *net) const {
    real_t result = 0.0;

    for(size_t i = 0; i < output.size(); i++) {
        real_t err = abs(net->get_output(i) - output[i]);

        if(err < 0.05) {
            err = 0.0;
        }

        result += err;
    }

    return result * weight;
}

//------------------------------
//---
//--- CLASS TestBattery
//---
//------------------------------
void TestBattery::add(const Test &test) {
    for(auto &step: test.steps) {
        max_err += step.weight * step.output.size();
    }
    tests.push_back(test);
}

TestBattery::EvalResult TestBattery::evaluate(Organism &org) const {
    Network *net = &org.net;
    real_t errorsum = 0.0;

    for(auto &test: tests) {
        for(auto &step: test.steps) {
            net->load_sensors(step.input);
            for(size_t i = 0; i < NACTIVATES_PER_INPUT; i++) {
                net->activate();
            }
            errorsum += step.err( net );
        }

        net->flush();
    }

    EvalResult result;
    result.fitness = 1.0 - errorsum/max_err;
    result.error = errorsum;

    return result;
}

//------------------------------
//---
//--- CLASS Experiment
//---
//------------------------------
map<string, Experiment *> *Experiment::experiments = nullptr;

Experiment *Experiment::get(const char *name) {
    if(!experiments) {
        experiments = new map<string, Experiment*>();
    }
    auto it = experiments->find(name);
    return it == experiments->end() ? nullptr : it->second;
}

vector<string> Experiment::get_names() {
    vector<string> result;
    if(experiments) {
        for(auto &kv: *experiments) {
            result.push_back(kv.first);
        }
    }
    return result;
}

Experiment::Experiment(const char *name) {
    this->name = name;
    if(get(name) != nullptr) {
        trap("Experiment already registered: " << name);
    }
    experiments->insert(make_pair(name, this));
}

Experiment::~Experiment() {
    experiments->erase(name);
    if(experiments->size() == 0) {
        delete experiments;
        experiments = nullptr;
    }
}

void Experiment::init_env() {
        const bool DELETE_NODES = true;
        const bool DELETE_LINKS = true;

        if(DELETE_NODES) {
            NEAT::mutate_delete_node_prob = 0.1 * NEAT::mutate_add_node_prob;
        }

        if(DELETE_LINKS) {
            NEAT::mutate_delete_link_prob = 0.1 * NEAT::mutate_add_link_prob;

            NEAT::mutate_toggle_enable_prob = 0.0;
            NEAT::mutate_gene_reenable_prob = 0.0;
        }

        NEAT::compat_threshold = 10.0;
}

void Experiment::print(Population *pop, int experiment_num, int geneneration) {
    ofstream out(get_fittest_path(experiment_num, geneneration));
    pop->get_fittest().write(out);
}

void Experiment::init() {
    init_env();

    vector<Test> tests = create_tests();

    //Validate tests
    {
        assert(tests.size() > 0);
        assert(tests[0].steps.size() > 0);

        ninputs = tests[0].steps[0].input.size();
        assert(ninputs > 0);

        noutputs = tests[0].steps[0].output.size();
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

    //Organize tests into batteries.
    for(Test &test: tests) {
        batteries[test.type].add(test);
    }
    assert(batteries.find(Test::Training) != batteries.end());

    cout << "=====================" << endl;
    cout << "===== BATTERIES =====" << endl;
    cout << "=====================" << endl;
    for(auto &kv: batteries) {
        cout << "---" << endl;
        cout << "---" << str(kv.first) << endl;
        cout << "---" << endl;

        for(Test &t: kv.second.tests) {
            for(Step s: t.steps) {
                for(real_t i: s.input) {
                    printf("%1.3f ", i);
                }
                printf("| ");
                for(real_t o: s.output) {
                    printf("%1.3f ", o);
                }
                printf(" ; weight=%f", s.weight);
                printf("\n");
            }
            printf("~~~.\n");
        }
    }
}

void Experiment::run(rng_t &rng, int gens) {
            
    int nsuccesses = 0;

    for(int expcount = 1; expcount <= NEAT::num_runs; expcount++) {
        mkdir( get_dir_path(expcount) );

        //Create a unique rng sequence for this experiment
        rng_t rng_exp(rng.integer());

        GenomeManager *genome_manager = GenomeManager::create();
        vector<unique_ptr<Genome>> genomes = 
            genome_manager->create_seed_generation(NEAT::pop_size,
                                                   rng_exp,
                                                   1,
                                                   ninputs,
                                                   noutputs,
                                                   ninputs);
        //Spawn the Population
        Population *pop = Population::create(rng_exp, genome_manager, genomes);
      
        bool success = false;
        int gen;
        for(gen = 1; !success && (gen <= gens); gen++) {
            cout << "Epoch " << gen << " . Experiment " << expcount << "/" << NEAT::num_runs << endl;	

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
                print(pop, expcount, gen);
        }

        print(pop, expcount, gen - 1);

        delete pop;
        delete genome_manager;
    }

    cout << "Failures: " << (NEAT::num_runs - nsuccesses) << " out of " << NEAT::num_runs << " runs" << endl;
}

bool Experiment::is_success(Organism *org) {
    return org->error <= 0.0000001;
}

void Experiment::evaluate(Population *pop) {
    TestBattery &battery = batteries[Test::Training];
    auto eval = [=] (Organism &org) {
        TestBattery::EvalResult result = battery.evaluate(org);
        org.fitness = result.fitness;
        org.error = result.error;
    };

    bool new_fittest = pop->evaluate(eval);
    Organism &fittest = pop->get_fittest();

    if(new_fittest) {
    }

    Genome::Stats gstats = fittest.genome->get_stats();
    cout << "fittest [" << fittest.population_index << "]"
         << ": fitness=" << fittest.fitness
         << ", error=" << fittest.error
         << ", nnodes=" << gstats.nnodes
         << ", nlinks=" << gstats.nlinks
         << endl;
}

