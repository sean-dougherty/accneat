#include "std.h" // Must be included first. Precompiled header with standard library includes.
#include "experiment.h"
#include "network.h"
#include "organismevaluator.h"
#include "population.h"
#include "stats.h"
#include "timer.h"
#include <assert.h>

using namespace NEAT;
using namespace std;

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

static const char *str(Test::Type t) {
    switch(t) {
    case Test::Training: return "Training";
    case Test::Fittest: return "Fittest";
    default: panic();
    }
}

static const char *str(Step::ErrType t) {
    switch(t) {
    case Step::Err_Delta: return "delta";
    case Step::Err_Binary: return "binary";
    default: panic();
    }
}

//------------------------------
//---
//--- CLASS Step
//---
//------------------------------
Step::Step(const vector<real_t> &input_,
           const vector<real_t> &output_,
           real_t weight_,
           ErrType err_type_)
    : input(input_)
    , output(output_)
    , weight(weight_)
    , err_type(err_type_) {

    if(err_type == Err_Binary) {
        for(auto x: output) {
            assert(x == 0.0 || x == 1.0);
        }
    }
}

real_t Step::evaluate(Organism &org) {
    Network *net = org.net.get();

    switch(err_type) {
    case Err_Delta: {
        real_t result = 0.0;

        for(size_t i = 0; i < output.size(); i++) {
            real_t err = abs(net->get_output(i) - output[i]);
            if(err < 0.05) {
                err = 0.0;
            }
            result += err;
        }

        return result * weight;
    } break;
    case Err_Binary: {
        size_t result = 0;

        for(size_t i = 0; i < output.size(); i++) {
            if( int(net->get_output(i) + 0.5) != int(output[i]) ) {
                result++;
            }
        }

        return real_t(result) * weight;
    } break;
    default:
        panic();
    }
}

//------------------------------
//---
//--- CLASS Test
//---
//------------------------------
Test::Test(const string &name_,
           const vector<Step> &steps_,
           Type type_)
    : name(name_), steps(steps_), type(type_) {
}

void Test::prepare(Organism &org, size_t istep) {
    Step &step = steps[istep];
    org.net->load_sensors(step.input, istep == 0);
}

real_t Test::evaluate(Organism &org, size_t istep) {
    return steps[istep].evaluate(org);
}

//------------------------------
//---
//--- CLASS TestBattery
//---
//------------------------------
TestBattery::TestBattery(const std::vector<Test> &tests_) : tests(tests_) {
    population_err.resize(env->pop_size);
    
    for(size_t itest = 0; itest < tests.size(); itest++) {
        for(size_t istep = 0; istep < tests[itest].steps.size(); istep++) {
            test_steps.push_back({itest, istep});
        }
    }

    max_err = 0.0;
    for(Test &test: tests) {
        for(Step &step: test.steps) {
            max_err += step.weight * step.output.size();
        }
    }
}

bool TestBattery::prepare_step(Organism &org, size_t istep) {
    if(istep >= test_steps.size())
        return false;
    else if(istep == 0)
        population_err[org.population_index] = 0.0;

    TestStep &step = test_steps[istep];
    tests[step.itest].prepare(org, step.istep);

    return true;
}

void TestBattery::evaluate_step(Organism &org, size_t istep) {
    TestStep &step = test_steps[istep];

    population_err[org.population_index] += tests[step.itest].evaluate(org, step.istep);
}

OrganismEvaluation TestBattery::get_evaluation(Organism &org) {
    OrganismEvaluation result;
    result.error = population_err[org.population_index];
    result.fitness = 1.0 - result.error/max_err;
    return result;
}

/*
void TestBattery::show_report(Organism &org) {
    vector<string> errant, perfect;
    for(auto &test: tests) {
        test.evaluate(org);
        if(test.err(org) != 0.0) {
            errant.push_back(test.name);
        } else {
            perfect.push_back(test.name);
        }
    }

    cout << str(tests.front().type) << ": ";

    if(tests.front().name == "") {
        cout << "perfect=" << (100 * real_t(perfect.size())/tests.size()) << "%" << endl;
    } else {
        cout << "perfect[" << perfect.size() << "]={";
        for(size_t i = 0; i < perfect.size(); i++) {
            if(i != 0) cout << ",";
            cout << perfect[i];
        }
        cout << "} ";

        cout << "errant[" << errant.size() << "]={";
        for(size_t i = 0; i < errant.size(); i++) {
            if(i != 0) cout << ",";
            cout << errant[i];
        }
        cout << "} ";

        cout << endl;
    }
}
*/

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
}

void Experiment::print(OrganismEvaluator *eval,
                       int experiment_num,
                       int geneneration) {
    ofstream out(get_fittest_path(experiment_num, geneneration));
    eval->get_fittest()->write(out);
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

    {
        map<Test::Type, vector<Test>> testmap;
        //Organize tests into batteries.
        for(Test &test: tests) {
            testmap[test.type].push_back(test);
        }
        assert( contains(testmap, Test::Training) );

        for(auto &kv: testmap) {
            batteries.emplace(kv.first, kv.second);
        }
    }

    cout << "=====================" << endl;
    cout << "===== BATTERIES =====" << endl;
    cout << "=====================" << endl;
    for(auto &kv: batteries) {
        cout << "---" << endl;
        cout << "---" << str(kv.first) << endl;
        cout << "---" << endl;

        for(Test &t: kv.second.tests) {
            printf("~~~ %s\n", t.name.c_str());
            for(Step s: t.steps) {
                for(real_t i: s.input) {
                    printf("%1.3f ", i);
                }
                printf("| ");
                for(real_t o: s.output) {
                    printf("%1.3f ", o);
                }
                printf(" ; weight=%f", s.weight);
                printf(", err_type=%s", str(s.err_type));
                printf("\n");
            }
        }
    }
}

void Experiment::run(rng_t &rng, int gens) {
            
    int nsuccesses = 0;
    vector<int> success_generations;
    vector<size_t> nnodes;
    vector<size_t> nlinks;
    vector<real_t> fitness;

    for(int expcount = 1; expcount <= env->num_runs; expcount++) {
        mkdir( get_dir_path(expcount) );

        //Create a unique rng sequence for this experiment
        rng_t rng_exp(rng.integer());

        env->genome_manager = GenomeManager::create();
        vector<unique_ptr<Genome>> genomes = 
            env->genome_manager->create_seed_generation(env->pop_size,
                                                        rng_exp,
                                                        1,
                                                        ninputs,
                                                        noutputs,
                                                        ninputs);
        //Spawn the Population
        Population *pop = Population::create(rng_exp, genomes);
        unique_ptr<OrganismEvaluator> eval = make_unique<OrganismEvaluator>(pop);
      
        bool success = false;
        int gen;
        for(gen = 1; !success && (gen <= gens); gen++) {
            cout << "Epoch " << gen << " . Experiment " << expcount << "/" << env->num_runs << endl;	

            static Timer timer("epoch");
            timer.start();

            if(gen != 1) {
                pop->next_generation();
            }

            evaluate(eval.get());

            if(is_success(eval->get_fittest())) {
                success = true;
                nsuccesses++;
            }

            timer.stop();
            Timer::report();

            //Don't print on success because we'll exit the loop and print then.
            if(!success && (gen % env->print_every == 0))
                print(eval.get(), expcount, gen);
        }

        if(success) {
            success_generations.push_back(gen);
        }
        {
            Organism *fittest = eval->get_fittest();
            Genome::Stats gstats = fittest->genome->get_stats();
            fitness.push_back(fittest->eval.fitness);
            nnodes.push_back(gstats.nnodes);
            nlinks.push_back(gstats.nlinks);
        }

        print(eval.get(), expcount, gen - 1);

        delete pop;
        delete env->genome_manager;
    }

    cout << "Failures: " << (env->num_runs - nsuccesses) << " out of " << env->num_runs << " runs" << endl;
    if(success_generations.size() > 0) {
        cout << "Success generations: " << stats(success_generations) << endl;
    }
    cout << "fitness stats: " << stats(fitness) << endl;
    cout << "nnodes stats: " << stats(nnodes) << endl;
    cout << "nlinks stats: " << stats(nlinks) << endl;
}

bool Experiment::is_success(Organism *org) {
    return org->eval.error <= 0.0000001;
}

void Experiment::evaluate(OrganismEvaluator *evaluator) {
    TestBattery &battery = batteries.find(Test::Training)->second;

    auto prepare_step = [&battery] (Organism &org, size_t istep) {
        return battery.prepare_step(org, istep);
    };

    auto eval_step = [&battery] (Organism &org, size_t istep) {
        battery.evaluate_step(org, istep);
    };

    auto eval_org  = [&battery] (Organism &org) {
        return battery.get_evaluation(org);
    };

    evaluator->evaluate(prepare_step,
                        eval_step,
                        eval_org);

    Organism *fittest = evaluator->get_fittest();
    Genome::Stats gstats = fittest->genome->get_stats();
    cout << "fittest [" << fittest->population_index << "]"
         << ": fitness=" << fittest->eval.fitness
         << ", error=" << fittest->eval.error
         << ", nnodes=" << gstats.nnodes
         << ", nlinks=" << gstats.nlinks
         << endl;
}

