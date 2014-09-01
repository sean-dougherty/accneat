#include "seq_experiment.h"

#include "organism.h"

#include <assert.h>
#include <omp.h>

#include <iostream>
#include <sstream>

using namespace NEAT;
using namespace std;

struct Step {
    vector<float> input;
    vector<float> output;

    double err(vector<NNode*> &netout,
               float **details_act,
               float **details_err) {
        assert(netout.size() == output.size());

        double result = 0.0;

        for(size_t i = 0; i < netout.size(); i++) {
            double diff = netout[i]->activation - output[i];
            double err = diff * diff;

            if(err < (0.05 * 0.05)) {
                err = 0.0;
            }

            result += err;

            **details_act = netout[i]->activation;
            **details_err = err;

            (*details_act)++;
            (*details_err)++;
        }

        return result;
    }
};

struct Test {
    vector<Step> steps;
    Test(const vector<Step> &steps_) : steps(steps_) {
        // Insert 1.0 for bias sensor
        for(auto &step: steps) {
            step.input.insert(step.input.begin(), 1.0f);
        }
    }
};

const float S = 1.0; // Signal
const float Q = 1.0; // Query
const float _ = 0.0; // Null

const float A = 0.0;
const float B = 0.5;
const float C = 1.0;

vector<Test> tests = {
    Test({
            {{S, _, A, A}, {_, _}},
            {{_, Q, _, _}, {A, A}}
        }),
    Test({
            {{S, _, A, B}, {_, _}},
            {{_, Q, _, _}, {A, B}}
        }),
    Test({
            {{S, _, A, C}, {_, _}},
            {{_, Q, _, _}, {A, C}}
        }),

    Test({
            {{S, _, B, A}, {_, _}},
            {{_, Q, _, _}, {B, A}}
        }),
    Test({
            {{S, _, B, B}, {_, _}},
            {{_, Q, _, _}, {B, B}}
        }),
    Test({
            {{S, _, B, C}, {_, _}},
            {{_, Q, _, _}, {B, C}}
        }),

    Test({
            {{S, _, C, A}, {_, _}},
            {{_, Q, _, _}, {C, A}}
        }),
    Test({
            {{S, _, C, B}, {_, _}},
            {{_, Q, _, _}, {C, B}}
        }),
    Test({
            {{S, _, C, C}, {_, _}},
            {{_, Q, _, _}, {C, C}}
        }),
};

const size_t nsteps = []() {
    size_t n = 0;
    for(auto &test: tests) n += test.steps.size();
    return n;
}();
const size_t nouts = []() {
    return nsteps * tests[0].steps[0].output.size();
}();

static float score(float errorsum) {
    return nouts - errorsum;
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
        for(auto &test: tests) {
            for(auto &step: test.steps) {
                double activation = activate(step.input);
                double err = step.err( net->outputs, &details_act, &details_err );
                errorsum += err;
                
            }

            net->flush();
        }
    }
 
    org->fitness = score(errorsum) / score(0.0);
    org->error=errorsum;

    org->winner = org->fitness >= 0.9999999;
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
    float details_act[n * nouts];
    float details_err[n * nouts];
#pragma omp parallel for
    for(size_t i = 0; i < n; i++) {
        Organism *org = pop->organisms[i];
        size_t details_offset = i * nouts;
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
        float *best_act = details_act + i_best * nouts;
        float *best_err = details_err + i_best * nouts;
        Organism *org = pop->organisms[i_best];
        
        printf("new best_fitness=%f; fitness=%f, errorsum=%f -- activation (err)\n",
               (float)best_fitness, (float)org->fitness, (float)org->error);

        for(auto &test: tests) {
            for(auto &step: test.steps) {
                for(auto &out: step.output) {
                    printf("%f (%f) ", *best_act++, *best_err++);
                }
                printf("\n");
            }
            printf("---\n");
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
