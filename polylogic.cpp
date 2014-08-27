#include "polylogic.h"

#include <assert.h>

#include <iostream>
#include <sstream>

using namespace NEAT;
using namespace std;

#define POLYLOGIC_THRESH_FALSE 0.3
#define POLYLOGIC_THRESH_TRUE 0.7

//Perform evolution on XOR, for gens generations
Population *polylogic_test(int gens) {
    Population *pop=0;
    Genome *start_genome;
    char curword[20];
    int id;

    ostringstream *fnamebuf;
    int gen;
 
    int evals[NEAT::num_runs];  //Hold records for each run
    int genes[NEAT::num_runs];
    int nodes[NEAT::num_runs];
    int winnernum;
    int winnergenes;
    int winnernodes;
    //For averaging
    int totalevals=0;
    int totalgenes=0;
    int totalnodes=0;
    int totalgens=0;
    int expcount;
    int samples;  //For averaging

    memset (evals, 0, NEAT::num_runs * sizeof(int));
    memset (genes, 0, NEAT::num_runs * sizeof(int));
    memset (nodes, 0, NEAT::num_runs * sizeof(int));

    ifstream iFile("polylogicstartgenes",ios::in);

    cout<<"START XOR TEST"<<endl;

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
            if (polylogic_epoch(pop,gen,temp,winnernum,winnergenes,winnernodes)) {
                //	if (polylogic_epoch(pop,gen,fnamebuf->str(),winnernum,winnergenes,winnernodes)) {
                //Collect Stats on end of experiment
                evals[expcount]=NEAT::pop_size*(gen-1)+winnernum;
                genes[expcount]=winnergenes;
                nodes[expcount]=winnernodes;
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

    cout<<"Failures: "<<(NEAT::num_runs-samples)<<" out of "<<NEAT::num_runs<<" runs"<<endl;
    cout<<"Average Generations: "<<double(totalgens)/expcount<<endl;
    cout<<"Average Nodes: "<<(samples>0 ? (double)totalnodes/samples : 0)<<endl;
    cout<<"Average Genes: "<<(samples>0 ? (double)totalgenes/samples : 0)<<endl;
    cout<<"Average Evals: "<<(samples>0 ? (double)totalevals/samples : 0)<<endl;

    return pop;

}

bool polylogic_evaluate(Organism *org) {
    Network *net;
    double out[4]; //The four outputs
    double this_out; //The current output
    int count;

    bool success;  //Check for successful activation
    int numnodes;  /* Used to figure out how many nodes
                      should be visited during activation */

    int net_depth; //The max depth of the network to be activated
    int relax; //Activates until relaxation

    struct Test {
        vector<float> input;
        float output;
    };

    vector<Test> tests = {
        {{1.0, 0.0, 0.0}, 0.0},
        {{1.0, 0.0, 1.0}, 1.0},
        {{1.0, 1.0, 0.0}, 1.0},
        {{1.0, 1.0, 1.0}, 0.0}
    };

    net=org->net;
    numnodes=((org->gnome)->nodes).size();

    net_depth=net->max_depth();

    for(size_t i = 0; i < tests.size(); i++) {
        Test &test = tests[i];

        net->load_sensors(test.input);

        //Relax net and get output
        success=net->activate();

        //use depth to ensure relaxation
        for (relax=0;relax<=net_depth;relax++) {
            success=net->activate();
        }

        out[i]=(*(net->outputs.begin()))->activation;

        net->flush();
    }

    if (success) {
        double errorsum = 0.0;
        for(size_t i = 0; i < tests.size(); i++) {
            errorsum += fabs(out[i] - tests[i].output);
        }
        org->fitness=pow((float(tests.size())-errorsum),2);
        org->error=errorsum;
    }
    else {
        //The network is flawed (shouldnt happen)
        org->fitness=0.001;
    }

    org->winner = (org->fitness / float(tests.size()*tests.size())) >= 0.95;

    return org->winner;
}

int polylogic_epoch(Population *pop,int generation,char *filename,int &winnernum,int &winnergenes,int &winnernodes) {
    vector<Organism*>::iterator curorg;
    vector<Species*>::iterator curspecies;
    //char cfilename[100];
    //strncpy( cfilename, filename.c_str(), 100 );

    //ofstream cfilename(filename.c_str());

    bool win=false;


    //Evaluate each organism on a test
    for(curorg=(pop->organisms).begin();curorg!=(pop->organisms).end();++curorg) {
        if (polylogic_evaluate(*curorg)) {
            win=true;
            winnernum=(*curorg)->gnome->genome_id;
            winnergenes=(*curorg)->gnome->extrons();
            winnernodes=((*curorg)->gnome->nodes).size();
            if (winnernodes==5) {
                //You could dump out optimal genomes here if desired
                //(*curorg)->gnome->print_to_filename("polylogic_optimal");
                //cout<<"DUMPED OPTIMAL"<<endl;
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
        for(curorg=(pop->organisms).begin();curorg!=(pop->organisms).end();++curorg) {
            if ((*curorg)->winner) {
                cout<<"WINNER IS #"<<((*curorg)->gnome)->genome_id<<endl;
                //Prints the winner to file
                //IMPORTANT: This causes generational file output!
                print_Genome_tofile((*curorg)->gnome,"polylogic_winner");
            }
        }
    
    }

    pop->epoch(generation);

    if (win) return 1;
    else return 0;

}
