#include "seq_experiment.h"

#include "organism.h"

#include <assert.h>
#include <omp.h>

#include <iostream>
#include <sstream>

using namespace NEAT;
using namespace std;

struct Frequency {
    const char *name;
    float minact;
    float maxact;

    float err(float x) {
        if(x > maxact) {
            float diff = (x - maxact) / (1.0f - maxact);
            return 0.5f + 0.5f * diff;
        } else if(x < minact) {
            float diff = (minact - x) / minact;
            return 0.5f + 0.5f * diff;
        } else {
            return 0.0f;
        }
    }
};

Frequency f_null = {"null", 0.0, 0.1};

vector<Frequency> f = {{"0", 0.45, 0.55}, {"1", 0.9, 1.0}};
           
struct Step {
    vector<float> input;
    Frequency output;
};

struct Test {
    vector<Step> steps;
    Test(const vector<Step> &steps_) : steps(steps_) {}
};

// First input is bias, not sensor.
vector<Test> tests = {
    //
    // Test first element
    // 
    Test({
            {{1.0, 1.0, 0.0, 0.0}, f_null}
            ,{{1.0, 1.0, 0.0, 0.0}, f_null}
            ,{{1.0, 0.0, 0.0, 0.0}, f_null}
            ,{{1.0, 0.0, 0.0, 0.5}, f[0]}
        }),
    Test({
            {{1.0, 1.0, 0.0, 0.0}, f_null}
            ,{{1.0, 0.0, 1.0, 0.0}, f_null}
            ,{{1.0, 0.0, 0.0, 0.0}, f_null}
            ,{{1.0, 0.0, 0.0, 0.5}, f[0]}
        }),
    Test({
            {{1.0, 0.0, 1.0, 0.0}, f_null}
            ,{{1.0, 1.0, 0.0, 0.0}, f_null}
            ,{{1.0, 0.0, 0.0, 0.0}, f_null}
            ,{{1.0, 0.0, 0.0, 0.5}, f[1]}
        }),
    Test({
            {{1.0, 0.0, 1.0, 0.0}, f_null}
            ,{{1.0, 0.0, 1.0, 0.0}, f_null}
            ,{{1.0, 0.0, 0.0, 0.0}, f_null}
            ,{{1.0, 0.0, 0.0, 0.5}, f[1]}
        }),

    //
    // Test second element
    // 
    Test({
            {{1.0, 1.0, 0.0, 0.0}, f_null}
            ,{{1.0, 1.0, 0.0, 0.0}, f_null}
            ,{{1.0, 0.0, 0.0, 0.0}, f_null}
            ,{{1.0, 0.0, 0.0, 1.0}, f[0]}
        }),
    Test({
            {{1.0, 1.0, 0.0, 0.0}, f_null}
            ,{{1.0, 0.0, 1.0, 0.0}, f_null}
            ,{{1.0, 0.0, 0.0, 0.0}, f_null}
            ,{{1.0, 0.0, 0.0, 1.0}, f[1]}
        }),
    Test({
            {{1.0, 0.0, 1.0, 0.0}, f_null}
            ,{{1.0, 1.0, 0.0, 0.0}, f_null}
            ,{{1.0, 0.0, 0.0, 0.0}, f_null}
            ,{{1.0, 0.0, 0.0, 1.0}, f[0]}
        }),
    Test({
            {{1.0, 0.0, 1.0, 0.0}, f_null}
            ,{{1.0, 0.0, 1.0, 0.0}, f_null}
            ,{{1.0, 0.0, 0.0, 0.0}, f_null}
            ,{{1.0, 0.0, 0.0, 1.0}, f[1]}
        })
};

const size_t nsteps = []() {
    size_t n = 0;
    for(auto &test: tests) n += test.steps.size();
    return n;
}();
const size_t nreps = 1;

static float score(float errorsum) {
    return nsteps * nreps - errorsum;
};


static bool evaluate(NEAT::Organism *org);
static int epoch(NEAT::Population *pop,
                 int generation,
                 char *filename,
                 int &winnernum,
                 int &winnergenes,
                 int &winnernodes,
                 int &winnerdepth);

//Perform evolution on SEQ_EXPERIMENT, for gens generations
Population *seq_experiment(int gens, char const *startgenes_path) {
    Population *pop=0;
    Genome *start_genome;
    char curword[20];
    int id;

    ostringstream *fnamebuf;
    int gen;
 
    int evals[NEAT::num_runs];
    int genes[NEAT::num_runs];
    int nodes[NEAT::num_runs];
    int depth[NEAT::num_runs];
    int winnernum;
    int winnergenes;
    int winnernodes;
    int winnerdepth;
    //For averaging
    int totalevals=0;
    int totalgenes=0;
    int totalnodes=0;
    int totaldepth=0;
    int totalgens=0;
    int expcount;
    int samples;  //For averaging

    memset (evals, 0, NEAT::num_runs * sizeof(int));
    memset (genes, 0, NEAT::num_runs * sizeof(int));
    memset (nodes, 0, NEAT::num_runs * sizeof(int));
    memset (depth, 0, NEAT::num_runs * sizeof(int));

    ifstream iFile(startgenes_path);

    cout<<"START SEQ_EXPERIMENT TEST"<<endl;

    cout<<"Reading in the start genome"<<endl;
    //Read in the start Genome
    iFile>>curword;
    iFile>>id;
    cout<<"Reading in Genome id "<<id<<endl;
    start_genome=new Genome(id,iFile);
    iFile.close();

    for(expcount=0;expcount<NEAT::num_runs;expcount++) {
        //Spawn the Population
        cout<<"Spawning Population off Genome2"<<endl;

        pop=new Population(start_genome,NEAT::pop_size);
      
        cout<<"Verifying Spawned Pop"<<endl;
        pop->verify();
      
        for (gen=1;gen<=gens;gen++) {
            cout<<"Epoch "<<gen<<endl;	

            //This is how to make a custom filename
            fnamebuf=new ostringstream();
            (*fnamebuf)<<"gen_"<<gen<<ends;  //needs end marker

#ifndef NO_SCREEN_OUT
            cout<<"name of fname: "<<fnamebuf->str()<<endl;
#endif

            char temp[50];
            sprintf (temp, "gen_%d", gen);

            //Check for success
            if (epoch(pop,gen,temp,winnernum,winnergenes,winnernodes, winnerdepth)) {
                //	if (seq_experiment_epoch(pop,gen,fnamebuf->str(),winnernum,winnergenes,winnernodes)) {
                //Collect Stats on end of experiment
                evals[expcount]=NEAT::pop_size*(gen-1)+winnernum;
                genes[expcount]=winnergenes;
                nodes[expcount]=winnernodes;
                depth[expcount]=winnerdepth;
                totalgens += gen;
                gen=gens;

            }
	
            //Clear output filename
            fnamebuf->clear();
            delete fnamebuf;
	
        }

        if (expcount<NEAT::num_runs-1) delete pop;
      
    }

    //Average and print stats
    cout<<"Nodes: "<<endl;
    for(expcount=0;expcount<NEAT::num_runs;expcount++) {
        cout<<nodes[expcount]<<endl;
        totalnodes+=nodes[expcount];
    }
    
    cout<<"Genes: "<<endl;
    for(expcount=0;expcount<NEAT::num_runs;expcount++) {
        cout<<genes[expcount]<<endl;
        totalgenes+=genes[expcount];
    }
    
    cout<<"Evals "<<endl;
    samples=0;
    for(expcount=0;expcount<NEAT::num_runs;expcount++) {
        cout<<evals[expcount]<<endl;
        if (evals[expcount]>0)
        {
            totalevals+=evals[expcount];
            samples++;
        }
    }

    cout<<"Depth: "<<endl;
    for(expcount=0;expcount<NEAT::num_runs;expcount++) {
        cout<<depth[expcount]<<endl;
        totaldepth+=depth[expcount];
    }
    
    cout<<"Failures: "<<(NEAT::num_runs-samples)<<" out of "<<NEAT::num_runs<<" runs"<<endl;
    cout<<"Average Generations: "<<double(totalgens)/expcount<<endl;
    cout<<"Average Nodes: "<<(samples>0 ? (double)totalnodes/samples : 0)<<endl;
    cout<<"Average Genes: "<<(samples>0 ? (double)totalgenes/samples : 0)<<endl;
    cout<<"Average Evals: "<<(samples>0 ? (double)totalevals/samples : 0)<<endl;
    cout<<"Average Depth: "<<(samples>0 ? (double)totaldepth/samples : 0)<<endl;

    return pop;

}

bool evaluate(Organism *org, float *details_act, float *details_err) {
    Network *net;
    int count;

    int numnodes;  /* Used to figure out how many nodes
                      should be visited during activation */
    int net_depth; //The max depth of the network to be activated


    net=org->net;
    numnodes=((org->gnome)->nodes).size();
    net_depth=net->max_depth();

    auto activate = [net, net_depth] (vector<float> &input) {
        net->load_sensors(input);

        //Relax net and get output
        net->activate();

        //use depth to ensure relaxation
        for(int relax=0; relax <= net_depth; relax++) {
            net->activate();
        }

        return (*(net->outputs.begin()))->activation;        
    };

    float errorsum = 0.0;

    {
        for(size_t rep = 0; rep < nreps; rep++) {
            for(auto &test: tests) {
                for(auto &step: test.steps) {
                    double activation = activate(step.input);
                    double err = step.output.err( activation );
                    errorsum += err;

                    *details_act = activation;
                    details_act++;
                    *details_err = err;
                    details_err++;
                }

                net->flush();
            }
        }
    }
 
    org->fitness = score(errorsum) / score(0.0);
    org->error=errorsum;

    org->winner = org->fitness >= 0.99999;
    if(org->winner) {
        cout << "FOUND A WINNER: " << org->fitness << endl;
        cout.flush();
    }

    return org->winner;
}

int epoch(Population *pop,
          int generation,
          char *filename,
          int &winnernum,
          int &winnergenes,
          int &winnernodes,
          int &winnerdepth) {

    static float best_fitness = 0.0f;

    vector<Species*>::iterator curspecies;

    bool win=false;
    //Evaluate each organism on a test

    bool best = false;
    size_t i_best;

    const size_t n = pop->organisms.size();
    float details_act[n * nreps * nsteps];
    float details_err[n * nreps * nsteps];
#pragma omp parallel for
    for(size_t i = 0; i < n; i++) {
        Organism *org = pop->organisms[i];
        size_t details_offset = i * nreps * nsteps;
        if (evaluate(org, details_act + details_offset, details_err + details_offset)) {
#pragma omp critical
            {
                win=true;
                winnernum=org->gnome->genome_id;
                winnergenes=org->gnome->extrons();
                winnernodes=org->gnome->nodes.size();
                winnerdepth=org->net->max_depth();
            }
        }

        if(org->fitness > best_fitness) {
#pragma omp critical
            if(org->fitness > best_fitness) {
                best = true;
                i_best = i;
                best_fitness = org->fitness;
            }
        }
    }

    if(best) {
        float *best_act = details_act + i_best * nsteps * nreps;
        float *best_err = details_err + i_best * nsteps * nreps;
        size_t idetail = 0;
        Organism *org = pop->organisms[i_best];
        
        printf("new best_fitness=%f; fitness=%f, errorsum=%f -- activation (err)\n",
               (float)best_fitness, (float)org->fitness, (float)org->error);
        for(size_t i = 0; i < nreps; i++) {
            for(size_t j = 0; j < nsteps; j++, idetail++) {
                printf("%f (%f) ", best_act[idetail], best_err[idetail]);

                if((idetail+1) % 4 == 0) {
                    printf("\n");
                }
            }
        }
    }
  
    //Average and max their fitnesses for dumping to file and snapshot
    for(curspecies=(pop->species).begin();curspecies!=(pop->species).end();++curspecies) {

        //This experiment control routine issues commands to collect ave
        //and max fitness, as opposed to having the snapshot do it, 
        //because this allows flexibility in terms of what time
        //to observe fitnesses at

        (*curspecies)->compute_average_fitness();
        (*curspecies)->compute_max_fitness();
    }

    //Take a snapshot of the population, so that it can be
    //visualized later on
    //if ((generation%1)==0)
    //  pop->snapshot();

    //Only print to file every print_every generations
    if  (win||
         ((generation%(NEAT::print_every))==0))
        pop->print_to_file_by_species(filename);


    if (win) {
        for(Organism *org: pop->organisms) {
            if(org->winner) {
                cout << "WINNER IS #" << org->gnome->genome_id << endl;
                //Prints the winner to file
                //IMPORTANT: This causes generational file output!
                print_Genome_tofile(org->gnome,"seq_experiment_winner");
            }
        }
    
    }

    pop->epoch(generation);

    if (win) return 1;
    else return 0;

}
