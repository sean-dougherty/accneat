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
#include "species.h"
#include "organism.h"
#include <cmath>
#include <iostream>

using namespace NEAT;
using std::vector;

Species::Species(int i) {
	id=i;
	age=1;
	ave_fitness=0.0;
	expected_offspring=0;
	novel=false;
	age_of_last_improvement=0;
	max_fitness=0;
	max_fitness_ever=0;
	obliterate=false;

	average_est=0;
}

Species::Species(int i,bool n) {
	id=i;
	age=1;
	ave_fitness=0.0;
	expected_offspring=0;
	novel=n;
	age_of_last_improvement=0;
	max_fitness=0;
	max_fitness_ever=0;
	obliterate=false;

	average_est=0;
}


Species::~Species() {
}

bool Species::rank() {
	//organisms.qsort(order_orgs);
    std::sort(organisms.begin(), organisms.end(), order_orgs);
	return true;
}

bool Species::add_Organism(Organism *o){
	organisms.push_back(o);
	return true;
}

Organism *Species::get_champ() {
	double champ_fitness=-1.0;
	Organism *thechamp = nullptr;

    for(Organism *org: organisms) {
        if(org->fitness > champ_fitness) {
            thechamp = org;
            champ_fitness = thechamp->fitness;
        }
    }

	return thechamp;
}

bool Species::remove_org(Organism *org) {
	std::vector<Organism*>::iterator curorg;

	curorg=organisms.begin();
	while((curorg!=organisms.end())&&
		((*curorg)!=org))
		++curorg;

	if (curorg==organisms.end()) {
		//cout<<"ALERT: Attempt to remove nonexistent Organism from Species"<<endl;
		return false;
	}
	else {
		organisms.erase(curorg);
		return true;
	}
}

void Species::remove_eliminated() {
    size_t n = 0;
    for(size_t i = 0; i < organisms.size(); i++) {
        Organism *org = organisms[i];
        if(!org->eliminate) {
            organisms[n++] = organisms[i];
        }
    }
    organisms.resize(n);    
}

void Species::remove_generation(int gen) {
    size_t n = 0;
    for(size_t i = 0; i < organisms.size(); i++) {
        Organism *org = organisms[i];
        if(org->generation != gen) {
            organisms[n++] = organisms[i];
        }
    }
    organisms.resize(n);    
}

Organism *Species::first() {
	return *(organisms.begin());
}

//Print Species to a file outFile
bool Species::print_to_file(std::ostream &outFile) {
    //Print a comment on the Species info
    outFile<<std::endl<<"/* Species #"<<id<<" : (Size "<<organisms.size()<<") (AF "<<ave_fitness<<") (Age "<<age<<")  */"<<std::endl<<std::endl;

    //Print all the Organisms' Genomes to the outFile
    for(Organism *org: organisms) {
        //Put the fitness for each organism in a comment
        outFile<<std::endl<<"/* Organism #"<<(org->genome).genome_id<<" Fitness: "<<org->fitness<<" Error: "<<org->error<<" */"<<std::endl;

        //If it is a winner, mark it in a comment
        if (org->winner) outFile<<"/* ##------$ WINNER "<<(org->genome).genome_id<<" SPECIES #"<<id<<" $------## */"<<std::endl;

        (org->genome).print_to_file(outFile);
    }

    return true;
}

void Species::adjust_fitness() {
	std::vector<Organism*>::iterator curorg;

	int num_parents;
	int count;

	int age_debt; 

	age_debt=(age-age_of_last_improvement+1)-NEAT::dropoff_age;

	if (age_debt==0) age_debt=1;

	for(curorg=organisms.begin();curorg!=organisms.end();++curorg) {

		//Remember the original fitness before it gets modified
		(*curorg)->orig_fitness=(*curorg)->fitness;

		//Make fitness decrease after a stagnation point dropoff_age
		//Added an if to keep species pristine until the dropoff point
		//obliterate is used in competitive coevolution to mark stagnation
		//by obliterating the worst species over a certain age
		if ((age_debt>=1)||obliterate) {

			//Possible graded dropoff
			//((*curorg)->fitness)=((*curorg)->fitness)*(-atan(age_debt));

			//Extreme penalty for a long period of stagnation (divide fitness by 100)
			((*curorg)->fitness)=((*curorg)->fitness)*0.01;
			//std::cout<<"OBLITERATE Species "<<id<<" of age "<<age<<std::endl;
			//std::cout<<"dropped fitness to "<<((*curorg)->fitness)<<std::endl;
		}

		//Give a fitness boost up to some young age (niching)
		//The age_significance parameter is a system parameter
		//  if it is 1, then young species get no fitness boost
		if (age<=10) ((*curorg)->fitness)=((*curorg)->fitness)*NEAT::age_significance; 

		//Do not allow negative fitness
		if (((*curorg)->fitness)<0.0) (*curorg)->fitness=0.0001; 

		//Share fitness with the species
		(*curorg)->fitness=((*curorg)->fitness)/(organisms.size());

	}

	//Sort the population and mark for death those after survival_thresh*pop_size
	//organisms.qsort(order_orgs);
	std::sort(organisms.begin(), organisms.end(), order_orgs);

	//Update age_of_last_improvement here
	if (((*(organisms.begin()))->orig_fitness)> 
	    max_fitness_ever) {
	  age_of_last_improvement=age;
	  max_fitness_ever=((*(organisms.begin()))->orig_fitness);
	}

	//Decide how many get to reproduce based on survival_thresh*pop_size
	//Adding 1.0 ensures that at least one will survive
	num_parents=(int) floor((NEAT::survival_thresh*((double) organisms.size()))+1.0);
	
	//Mark for death those who are ranked too low to be parents
	curorg=organisms.begin();
	(*curorg)->champion=true;  //Mark the champ as such
	for(count=1;count<=num_parents;count++) {
	  if (curorg!=organisms.end())
	    ++curorg;
	}
	while(curorg!=organisms.end()) {
	  (*curorg)->eliminate=true;  //Mark for elimination
	  //std::std::cout<<"marked org # "<<(*curorg)->gnome->genome_id<<" fitness = "<<(*curorg)->fitness<<std::std::endl;
	  ++curorg;
	}             

}

double Species::compute_average_fitness() {
	std::vector<Organism*>::iterator curorg;

	double total=0.0;

	//int pause; //DEBUG: Remove

	for(curorg=organisms.begin();curorg!=organisms.end();++curorg) {
		total+=(*curorg)->fitness;
		//std::cout<<"new total "<<total<<std::endl; //DEBUG: Remove
	}

	ave_fitness=total/(organisms.size());

	//DEBUG: Remove
	//std::cout<<"average of "<<(organisms.size())<<" organisms: "<<ave_fitness<<std::endl;
	//cin>>pause;

	return ave_fitness;

}

double Species::compute_max_fitness() {
	double max=0.0;
	std::vector<Organism*>::iterator curorg;

	for(curorg=organisms.begin();curorg!=organisms.end();++curorg) {
		if (((*curorg)->fitness)>max)
			max=(*curorg)->fitness;
	}

	max_fitness=max;

	return max;
}

double Species::count_offspring(double skim) {
	std::vector<Organism*>::iterator curorg;
	int e_o_intpart;  //The floor of an organism's expected offspring
	double e_o_fracpart; //Expected offspring fractional part
	double skim_intpart;  //The whole offspring in the skim

	expected_offspring=0;

	for(curorg=organisms.begin();curorg!=organisms.end();++curorg) {
		e_o_intpart=(int) floor((*curorg)->expected_offspring);
		e_o_fracpart=fmod((*curorg)->expected_offspring,1.0);

		expected_offspring+=e_o_intpart;

		//Skim off the fractional offspring
		skim+=e_o_fracpart;

		//NOTE:  Some precision is lost by computer
		//       Must be remedied later
		if (skim>1.0) {
			skim_intpart=floor(skim);
			expected_offspring+=(int) skim_intpart;
			skim-=skim_intpart;
		}
	}

	return skim;

}

static Organism *get_random(rng_t &rng, Species *thiz, const vector<Species *> &sorted_species) {
    Species *result = thiz;
    for(int i = 0; (result == thiz) && (i < 5); i++) {
        double randmult = std::min(1.0, rng.gauss() / 4);
        int randspeciesnum = std::max(0, (int)floor((randmult*(sorted_species.size()-1.0))+0.5));
        result = sorted_species[randspeciesnum];
    }

    return result->first();
}

void Species::reproduce(vector<Organism> &pop_orgs,
                        size_t &iorg,
                        int generation,
                        Population *pop,
                        vector<Species*> &sorted_species) {
	Organism *mom = nullptr; //Parent Organisms
	Organism *dad = nullptr;

    bool champ_done=false; //Flag the preservation of the champion
	Organism *thechamp = nullptr;

	//Check for a mistake
	if ((expected_offspring>0) && (organisms.size()==0)) {
        trap("expected > 0 && norgs = 0");
    }

    thechamp = organisms[0];

    //Create the designated number of offspring for the Species
    //one at a time
    for(int count=0; count < expected_offspring; count++) {
        Organism *baby = &pop_orgs[iorg++];
        baby->init(0.0, generation);

        Genome *new_genome = &baby->genome;  //For holding baby's genes
        new_genome->reset(iorg+1);

        //If we have a super_champ (Population champion), finish off some special clones
        if ((thechamp->super_champ_offspring) > 0) {
            mom=thechamp;
            mom->genome.duplicate_into(*new_genome, count);

            //Most superchamp offspring will have their connection weights mutated only
            //The last offspring will be an exact duplicate of this super_champ
            //Note: Superchamp offspring only occur with stolen babies!
            //      Settings used for published experiments did not use this
            if ((thechamp->super_champ_offspring) > 1) {
                if ((rng.prob()<0.8)||
                    (NEAT::mutate_add_link_prob==0.0)) 
                    //ABOVE LINE IS FOR:
                    //Make sure no links get added when the system has link adding disabled
                    new_genome->mutate_link_weights(NEAT::weight_mut_power,1.0,GAUSSIAN);
                else {
                    //Sometimes we add a link to a superchamp
                    new_genome->mutate_add_link(pop->innovations,pop->cur_innov_num,NEAT::newlink_tries);
                }
            }

            thechamp->super_champ_offspring--;
        }
        //If we have a Species champion, just clone it 
        else if ((!champ_done)&&
                 (expected_offspring>5)) {

            mom=thechamp; //Mom is the champ
            mom->genome.duplicate_into(*new_genome, count);

            champ_done=true;
        }
        //First, decide whether to mate or mutate
        //If there is only one organism in the pool, then always mutate
        else if( (rng.prob() < NEAT::mutate_only_prob) || (organisms.size() == 1) ) {

            //Choose the random parent
            mom = rng.element(organisms);
            mom->genome.duplicate_into(*new_genome, count);

            //Do the mutation depending on probabilities of 
            //various mutations

            if (rng.prob()<NEAT::mutate_add_node_prob) {
                new_genome->mutate_add_node(pop->innovations,pop->cur_node_id,pop->cur_innov_num);
            }
            else if (rng.prob()<NEAT::mutate_add_link_prob) {
                new_genome->mutate_add_link(pop->innovations,pop->cur_innov_num,NEAT::newlink_tries);
            }
            //NOTE:  A link CANNOT be added directly after a node was added because the phenotype
            //       will not be appropriately altered to reflect the change
            else {
                //If we didn't do a structural mutation, we do the other kinds

                if (rng.prob()<NEAT::mutate_random_trait_prob) {
                    new_genome->mutate_random_trait();
                }
                if (rng.prob()<NEAT::mutate_link_trait_prob) {
                    new_genome->mutate_link_trait(1);
                }
                if (rng.prob()<NEAT::mutate_node_trait_prob) {
                    new_genome->mutate_node_trait(1);
                }
                if (rng.prob()<NEAT::mutate_link_weights_prob) {
                    new_genome->mutate_link_weights(NEAT::weight_mut_power,1.0,GAUSSIAN);
                }
                if (rng.prob()<NEAT::mutate_toggle_enable_prob) {
                    new_genome->mutate_toggle_enable(1);
                }
                if (rng.prob()<NEAT::mutate_gene_reenable_prob) {
                    new_genome->mutate_gene_reenable();
                }
            }

        }

        //Otherwise we should mate 
        else {

            //Choose the random mom
            mom = rng.element(organisms);

            //Choose random dad
            if ((rng.prob()>NEAT::interspecies_mate_rate)) {
                //Mate within Species
                dad = rng.element(organisms);
            } else {
                dad = get_random(rng, this, sorted_species);
            }

            //Perform mating based on probabilities of differrent mating types
            if (rng.prob()<NEAT::mate_multipoint_prob) { 
                mom->genome.mate_multipoint(&dad->genome,
                                            new_genome,
                                            count,
                                            mom->orig_fitness,
                                            dad->orig_fitness);
            }
            else if (rng.prob()<(NEAT::mate_multipoint_avg_prob/(NEAT::mate_multipoint_avg_prob+NEAT::mate_singlepoint_prob))) {
                mom->genome.mate_multipoint_avg(&dad->genome,
                                                new_genome,
                                                count,
                                                mom->orig_fitness,
                                                dad->orig_fitness);
            }
            else {
                // todo: catch non-zero probability at time of parsing. completely elim this
                // from code.
                std::cerr << "singlepoint mating no longer supported" << std::endl;
            }

            //Determine whether to mutate the baby's Genome
            //This is done randomly or if the mom and dad are the same organism
            if ((rng.prob()>NEAT::mate_only_prob)||
                ((dad->genome).genome_id==(mom->genome).genome_id)||
                (((dad->genome).compatibility(&mom->genome))==0.0))
            {

                //Do the mutation depending on probabilities of 
                //various mutations
                if (rng.prob()<NEAT::mutate_add_node_prob) {
                    new_genome->mutate_add_node(pop->innovations,pop->cur_node_id,pop->cur_innov_num);
                    //  std::cout<<"mutate_add_node: "<<new_genome<<std::endl;
                } else if (rng.prob()<NEAT::mutate_add_link_prob) {
                    new_genome->mutate_add_link(pop->innovations,pop->cur_innov_num,NEAT::newlink_tries);
                } else {
                    //Only do other mutations when not doing sturctural mutations

                    if (rng.prob()<NEAT::mutate_random_trait_prob) {
                        new_genome->mutate_random_trait();
                    }
                    if (rng.prob()<NEAT::mutate_link_trait_prob) {
                        new_genome->mutate_link_trait(1);
                    }
                    if (rng.prob()<NEAT::mutate_node_trait_prob) {
                        new_genome->mutate_node_trait(1);
                    }
                    if (rng.prob()<NEAT::mutate_link_weights_prob) {
                        new_genome->mutate_link_weights(NEAT::weight_mut_power,1.0,GAUSSIAN);
                    }
                    if (rng.prob()<NEAT::mutate_toggle_enable_prob) {
                        new_genome->mutate_toggle_enable(1);
                    }
                    if (rng.prob()<NEAT::mutate_gene_reenable_prob) {
                        new_genome->mutate_gene_reenable(); 
                    }
                }
            }
        }

        baby->create_phenotype();
    }
}

bool NEAT::order_species(Species *x, Species *y) { 
	return (((*((x->organisms).begin()))->orig_fitness) > ((*((y->organisms).begin()))->orig_fitness));
}

bool NEAT::order_new_species(Species *x, Species *y) {
	return (x->compute_max_fitness() > y->compute_max_fitness());
}


