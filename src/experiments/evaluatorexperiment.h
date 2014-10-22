#pragma once

#include "experiment.h"
#include "genomemanager.h"
#include "network.h"
#include "organism.h"
#include "population.h"
#include "stats.h"
#include "timer.h"
#include "util.h"

namespace NEAT {

//------------------------------
//---
//--- UTIL
//---
//------------------------------
    inline std::string get_dir_path(int experiment_num) {
        char buf[1024];
        sprintf(buf, "./experiment_%d", experiment_num);
        return buf;
    }

    inline std::string get_fittest_path(int experiment_num, int generation) {
        char buf[1024];
        sprintf(buf, "%s/fittest_%d",
                get_dir_path(experiment_num).c_str(),
                generation);
        return buf;
    }


//------------------------------
//---
//--- CLASS EvaluatorExperiment
//---
//------------------------------
    template<typename Evaluator>
    class EvaluatorExperiment : public Experiment {
    protected:
        NetworkExecutor<Evaluator> *network_executor;

        EvaluatorExperiment(const char *name)
            : Experiment(name) {
            network_executor = nullptr;
        }

        virtual ~EvaluatorExperiment() {
            delete network_executor;
        }

        virtual void init_env() {}
        virtual void init_experiment() = 0;

        virtual bool is_success(Organism *org) {
            return org->eval.error <= 0.0000001;
        }

        size_t ninputs;
        size_t noutputs;
    public:
        virtual void run(class rng_t &rng, int gens) override {
            using namespace std;

            init_env();

            network_executor = create_network_executor<Evaluator>();

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

                fittest = nullptr;
                env->genome_manager = GenomeManager::create();
                vector<unique_ptr<Genome>> genomes = 
                    env->genome_manager->create_seed_generation(env->pop_size,
                                                                rng_exp,
                                                                1,
                                                                ninputs,
                                                                noutputs,
                                                                ninputs);
                //Spawn the Population
                pop = Population::create(rng_exp, genomes);
      
                bool success = false;
                int gen;
                for(gen = 1; !success && (gen <= gens); gen++) {
                    cout << "Epoch " << gen << " . Experiment " << expcount << "/" << env->num_runs << endl;	

                    static Timer timer("epoch");
                    timer.start();

                    if(gen != 1) {
                        pop->next_generation();
                    }

                    evaluate();

                    if(is_success(fittest.get())) {
                        success = true;
                        nsuccesses++;
                    }

                    timer.stop();
                    Timer::report();

                    //Don't print on success because we'll exit the loop and print then.
                    if(!success && (gen % env->print_every == 0))
                        print(expcount, gen);
                }

                if(success) {
                    success_generations.push_back(gen);
                }
                {
                    Genome::Stats gstats = fittest->genome->get_stats();
                    fitness.push_back(fittest->eval.fitness);
                    nnodes.push_back(gstats.nnodes);
                    nlinks.push_back(gstats.nlinks);
                }

                print(expcount, gen - 1);

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

    private:
        void print(int experiment_num,
                   int generation) {
            using namespace std;

            ofstream out(get_fittest_path(experiment_num, generation));
            fittest->write(out);
        }

        void evaluate() {
            using namespace std;

            static Timer timer("evaluate");
            timer.start();

            size_t norgs = pop->size();
            Network *nets[norgs];
            for(size_t i = 0; i < norgs; i++) {
                nets[i] = pop->get(i)->net.get();
            }
            OrganismEvaluation evaluations[norgs];

            network_executor->execute(nets, evaluations, norgs);

            Organism *best = nullptr;
            for(size_t i = 0; i < norgs; i++) {
                Organism *org = pop->get(i);
                org->eval = evaluations[i];
                if( !best || (org->eval.fitness > best->eval.fitness) ) {
                    best = org;
                }
            }

            timer.stop();

            if(!fittest || (best->eval.fitness > fittest->eval.fitness)) {
                fittest = pop->make_copy(best->population_index);
            }

            Genome::Stats gstats = fittest->genome->get_stats();
            cout << "fittest [" << fittest->population_index << "]"
                 << ": fitness=" << fittest->eval.fitness
                 << ", error=" << fittest->eval.error
                 << ", nnodes=" << gstats.nnodes
                 << ", nlinks=" << gstats.nlinks
                 << endl;
        }

        class Population *pop;
        std::unique_ptr<Organism> fittest;
    };

}
