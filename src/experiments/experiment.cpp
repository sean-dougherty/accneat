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

void Experiment::print(OrganismEvaluator *eval,
                       int experiment_num,
                       int geneneration) {
    ofstream out(get_fittest_path(experiment_num, geneneration));
    eval->get_fittest()->write(out);
}

void Experiment::run(rng_t &rng, int gens) {
    init_env();
    init_experiment();
            
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
    auto process_output = [this] (Network &net, size_t istep) {
        this->process_output(net, istep);
    };

    auto eval_org  = [this] (Organism &org) {
        return evaluate(org);
    };

    evaluator->evaluate(get_sensors(),
                        process_output,
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
