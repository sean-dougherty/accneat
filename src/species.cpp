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

#include "genomemanager.h"
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

bool Species::add_Organism(SpeciesOrganism *o){
	organisms.push_back(o);
	return true;
}

void Species::remove_eliminated() {
    erase_if(organisms, [](SpeciesOrganism *org) {return org->eliminate;});
}

void Species::remove_generation(int gen) {
    erase_if(organisms, [gen](SpeciesOrganism *org) {return org->generation == gen;});
}

SpeciesOrganism *Species::first() {
	return *(organisms.begin());
}

//Print Species to a file outFile
bool Species::print_to_file(std::ostream &outFile) {
    //Print a comment on the Species info
    outFile<<std::endl<<"/* Species #"<<id<<" : (Size "<<organisms.size()<<") (AF "<<ave_fitness<<") (Age "<<age<<")  */"<<std::endl<<std::endl;

    //Print all the Organisms' Genomes to the outFile
    for(SpeciesOrganism *org: organisms) {
        //Put the fitness for each organism in a comment
        outFile<<std::endl<<"/* Organism #"<<org->genome->genome_id<<" Fitness: "<<org->fitness<<" Error: "<<org->error<<" */"<<std::endl;

        org->genome->print(outFile);
    }

    return true;
}

void Species::adjust_fitness() {
	std::vector<SpeciesOrganism*>::iterator curorg;

	int num_parents;
	int count;

	int age_debt; 

	age_debt=(age-age_of_last_improvement+1)-NEAT::dropoff_age;

	if (age_debt==0) age_debt=1;

	for(curorg=organisms.begin();curorg!=organisms.end();++curorg) {

		//Remember the original fitness before it gets modified
        (*curorg)->adjusted_fitness = (*curorg)->fitness;

		//Make fitness decrease after a stagnation point dropoff_age
		//Added an if to keep species pristine until the dropoff point
		//obliterate is used in competitive coevolution to mark stagnation
		//by obliterating the worst species over a certain age
		if ((age_debt>=1)||obliterate) {
			//Extreme penalty for a long period of stagnation (divide fitness by 100)
			((*curorg)->adjusted_fitness)=((*curorg)->adjusted_fitness)*0.01;
		}

		//Give a fitness boost up to some young age (niching)
		//The age_significance parameter is a system parameter
		//  if it is 1, then young species get no fitness boost
		if (age<=10) ((*curorg)->adjusted_fitness)=((*curorg)->adjusted_fitness)*NEAT::age_significance; 

		//Do not allow negative fitness
		if (((*curorg)->adjusted_fitness)<0.0) (*curorg)->adjusted_fitness=0.0001; 

		//Share fitness with the species
		(*curorg)->adjusted_fitness=((*curorg)->adjusted_fitness)/(organisms.size());
	}

	//Sort the population and mark for death those after survival_thresh*pop_size
	std::sort(organisms.begin(), organisms.end(),
              [] (SpeciesOrganism *x, SpeciesOrganism *y) {
                  return x->adjusted_fitness > y->adjusted_fitness;
              });

	//Update age_of_last_improvement here
	if (((*(organisms.begin()))->fitness)> 
	    max_fitness_ever) {
	  age_of_last_improvement=age;
	  max_fitness_ever=((*(organisms.begin()))->fitness);
	}

	//Decide how many get to reproduce based on survival_thresh*pop_size
	//Adding 1.0 ensures that at least one will survive
	num_parents=(int) floor((NEAT::survival_thresh*((real_t) organisms.size()))+1.0);
	
	//Mark for death those who are ranked too low to be parents
	curorg=organisms.begin();
	(*curorg)->champion=true;  //Mark the champ as such
	for(count=1;count<=num_parents;count++) {
	  if (curorg!=organisms.end())
	    ++curorg;
	}
	while(curorg!=organisms.end()) {
	  (*curorg)->eliminate=true;  //Mark for elimination
	  ++curorg;
	}             

}

real_t Species::compute_average_fitness() {
	std::vector<SpeciesOrganism*>::iterator curorg;

	real_t total=0.0;

	for(curorg=organisms.begin();curorg!=organisms.end();++curorg) {
		total+=(*curorg)->fitness;
	}

	ave_fitness=total/(organisms.size());

	return ave_fitness;

}

real_t Species::compute_max_fitness() {
	real_t max=0.0;
	std::vector<SpeciesOrganism*>::iterator curorg;

	for(curorg=organisms.begin();curorg!=organisms.end();++curorg) {
		if (((*curorg)->fitness)>max)
			max=(*curorg)->fitness;
	}

	max_fitness=max;

	return max;
}

real_t Species::count_offspring(real_t skim) {
	std::vector<SpeciesOrganism*>::iterator curorg;
	int e_o_intpart;  //The floor of an organism's expected offspring
	real_t e_o_fracpart; //Expected offspring fractional part
	real_t skim_intpart;  //The whole offspring in the skim

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

static SpeciesOrganism *get_random(rng_t &rng, Species *thiz, const vector<Species *> &sorted_species) {
    Species *result = thiz;
    for(int i = 0; (result == thiz) && (i < 5); i++) {
        real_t randmult = std::min(real_t(1.0), rng.gauss() / 4);
        int randspeciesnum = std::max(0, (int)floor((randmult*(sorted_species.size()-1.0))+0.5));
        result = sorted_species[randspeciesnum];
    }

    return result->first();
}

//todo: this method better belongs in the population class.
void Species::reproduce(int ioffspring,
                        SpeciesOrganism &baby,
                        GenomeManager *genome_manager,
                        vector<Species*> &sorted_species) {
    SpeciesOrganism *thechamp = organisms[0];
    Genome &new_genome = *baby.genome;  //For holding baby's genes
    rng_t &rng = baby.genome->rng;

    //If we have a super_champ (SpeciesPopulation champion), finish off some special clones
    if( ioffspring < thechamp->super_champ_offspring ) {
        genome_manager->clone(*thechamp->genome, new_genome);

        //Most superchamp offspring will have their connection weights mutated only
        //The last offspring will be an exact duplicate of this super_champ
        //Note: Superchamp offspring only occur with stolen babies!
        //      Settings used for published experiments did not use this
        if( ioffspring < (thechamp->super_champ_offspring - 1) ) {
            if ( (rng.prob() < 0.8)|| (NEAT::mutate_add_link_prob == 0.0)) {
                genome_manager->mutate(new_genome,
                                       GenomeManager::MUTATE_OP_WEIGHTS);
            } else {
                genome_manager->mutate(new_genome,
                                       GenomeManager::MUTATE_OP_STRUCTURE);
            }
        }

    } else if( (ioffspring == thechamp->super_champ_offspring) && (expected_offspring > 5) ) {

        //Clone the species champion
        genome_manager->clone(*thechamp->genome, new_genome);

    } else if( (rng.prob() < NEAT::mutate_only_prob) || (organisms.size() == 1) ) {

        //Clone a random parent
        genome_manager->clone(*rng.element(organisms)->genome,
                              new_genome);
        genome_manager->mutate(new_genome);

        //Otherwise we should mate 
    } else {
        //Choose the random mom
        SpeciesOrganism *mom = rng.element(organisms);

        //Choose random dad
        SpeciesOrganism *dad;
        if ((rng.prob() > NEAT::interspecies_mate_rate)) {
            //Mate within Species
            dad = rng.element(organisms);
        } else {
            dad = get_random(rng, this, sorted_species);
        }

        genome_manager->mate(*mom->genome,
                             *dad->genome,
                             new_genome,
                             mom->fitness,
                             dad->fitness);
    }
}

bool NEAT::order_species(Species *x, Species *y) { 
	return (((*((x->organisms).begin()))->fitness) > ((*((y->organisms).begin()))->fitness));
}

bool NEAT::order_new_species(Species *x, Species *y) {
	return (x->compute_max_fitness() > y->compute_max_fitness());
}


