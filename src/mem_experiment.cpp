#include "mem_experiment.h"

#include "organism.h"

#include <assert.h>

#include <iostream>
#include <sstream>

using namespace NEAT;
using namespace std;

static bool evaluate(NEAT::Organism *org);
static int epoch(NEAT::Population *pop,
                 int generation,
                 char *filename,
                 int &winnernum,
                 int &winnergenes,
                 int &winnernodes,
                 int &winnerdepth);

//Perform evolution on MEM_EXPERIMENT, for gens generations
Population *mem_experiment(int gens) {
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

    ifstream iFile("startgenes/mem_experiment",ios::in);

    cout<<"START MEM_EXPERIMENT TEST"<<endl;

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
                //	if (mem_experiment_epoch(pop,gen,fnamebuf->str(),winnernum,winnergenes,winnernodes)) {
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

bool evaluate(Organism *org) {
    Network *net;
    int count;

    int numnodes;  /* Used to figure out how many nodes
                      should be visited during activation */

    int net_depth; //The max depth of the network to be activated

    struct Test {
        vector<float> input;
        float output;
    };

    // First input is bias, not sensor.
    vector<Test> tests = {
        {{1.0, 0.0, 1.0, 0.0}, 0.5},
        {{1.0, 1.0, 0.0, 0.0}, 1.0}
    };
    vector<float> input_delay =
        {1.0, 0.0, 0.0, 0.0};
    vector<float> input_query =
        {1.0, 0.0, 0.0, 1.0};

    net=org->net;
    numnodes=((org->gnome)->nodes).size();
    net_depth=net->max_depth();

    double errorsum = 0.0;

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

    auto err = [] (float expected, float actual) {
        float diff = abs(expected - actual);
        if(diff < 0.01) {
            diff = 0.0;
        }
        return diff;
    };

    for(size_t i = 0; i < tests.size(); i++) {
        Test &test = tests[i];

        errorsum += err(0.0, activate(test.input));
        errorsum += err(0.0, activate(input_delay));
        errorsum += err(test.output, activate(input_query));

        net->flush();
    }

    auto score = [tests] (float errorsum) {
        return pow(tests.size() * 3 - errorsum, 2);
    };

    org->fitness = score(errorsum) / score(0.0);
    org->error=errorsum;

    org->winner = org->fitness >= 0.99999;
    return org->winner;
}

int epoch(Population *pop,
          int generation,
          char *filename,
          int &winnernum,
          int &winnergenes,
          int &winnernodes,
          int &winnerdepth) {

    vector<Organism*>::iterator curorg;
    vector<Species*>::iterator curspecies;

    bool win=false;


    //Evaluate each organism on a test
    for(curorg=(pop->organisms).begin();curorg!=(pop->organisms).end();++curorg) {
        if (evaluate(*curorg)) {
            win=true;
            winnernum=(*curorg)->gnome->genome_id;
            winnergenes=(*curorg)->gnome->extrons();
            winnernodes=((*curorg)->gnome->nodes).size();
            winnerdepth=(*curorg)->net->max_depth();
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
        for(curorg=(pop->organisms).begin();curorg!=(pop->organisms).end();++curorg) {
            if ((*curorg)->winner) {
                cout<<"WINNER IS #"<<((*curorg)->gnome)->genome_id<<endl;
                //Prints the winner to file
                //IMPORTANT: This causes generational file output!
                print_Genome_tofile((*curorg)->gnome,"mem_experiment_winner");
            }
        }
    
    }

    pop->epoch(generation);

    if (win) return 1;
    else return 0;

}
