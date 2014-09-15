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
#include "population.h"
#include "organism.h"
#include "timer.h"
#include <iostream>
#include <sstream>
#include <fstream>

#include <assert.h>
#include <omp.h>

using namespace NEAT;
using namespace std;

Population::OrganismsBuffer::OrganismsBuffer(rng_t &rng, size_t n)
    : _n(n) {
    _a.resize(n);
    _b.resize(n);
    _curr = &_a;

    int seed = 0;
    for(auto &org: _a)
        org.genome.rng.seed(seed++);
    for(auto &org: _b)
        org.genome.rng.seed(seed++);
}

size_t Population::OrganismsBuffer::size(){
    return _n;
}

vector<Organism> &Population::OrganismsBuffer::curr() {
    return *_curr;
}

void Population::OrganismsBuffer::swap() {
    if(_curr == &_a) {_curr = &_b;} else {_curr = &_a; }
    assert( _curr->size() == _n );
}

Population::Population(rng_t &rng, int size)
    : orgs(rng, size)
    , winnergen(0)
    , highest_fitness(0.0)
    , highest_last_changed(0) {
}

Population::Population(rng_t &rng, Genome *g,int size)
    : Population(rng, size) {

	spawn(g);
}

Population::Population(rng_t &rng, Genome *g,int size, float power)
    : Population(rng, size) {
	clone(g, power);
}

Population::~Population() {
	std::vector<Species*>::iterator curspec;
	std::vector<Organism*>::iterator curorg;

	if (species.begin()!=species.end()) {
		for(curspec=species.begin();curspec!=species.end();++curspec) {
			delete (*curspec);
		}
	}
}

void Population::verify() {
    for(auto &org: orgs.curr())
        org.genome.verify();
} 

bool Population::clone(Genome *g, float power) {

    //Create an exact clone.
    {
        Organism &org = orgs.curr()[0];
        g->duplicate_into(org.genome, 1);
        org.create_phenotype();
    }
	
	//Create copies of the Genome with perturbed linkweights
	for(size_t i = 1; i < size(); i++) {
        Organism &org = orgs.curr()[i];
        
        g->duplicate_into(org.genome, i+1);
		if(power>0)
			org.genome.mutate_link_weights(power, 1.0, GAUSSIAN);
		
        org.genome.randomize_traits();
        org.create_phenotype();
	}

	//Keep a record of the innovation and node number we are on
    innovations.init(orgs.curr().back().genome.get_last_node_id(),
                     orgs.curr().back().genome.get_last_gene_innovnum());

	//Separate the new Population into species
	speciate();

	return true;
}

bool Population::spawn(Genome *g) {
    for(size_t i = 0; i < size(); i++) {
        Organism &org = orgs.curr()[i];
        g->duplicate_into(org.genome, i+1); 
		org.genome.mutate_link_weights(1.0,1.0,COLDGAUSSIAN);
		org.genome.randomize_traits();
        org.create_phenotype();
	}

	//Keep a record of the innovation and node number we are on
    innovations.init(orgs.curr().back().genome.get_last_node_id(),
                     orgs.curr().back().genome.get_last_gene_innovnum());

	//Separate the new Population into species
	speciate();

	return true;
}

bool Population::speciate() {
    last_species = 0;
    for(Organism &org: orgs.curr()) {
        assert(org.species == nullptr);
        for(Species *s: species) {
            if( org.genome.compatibility(&s->first()->genome) < NEAT::compat_threshold ) {
                org.species = s;
                break;
            }
        }
        if(!org.species) {
            Species *s = new Species(++last_species);
            species.push_back(s);
            org.species = s;
        }
        org.species->add_Organism(&org);
    }
    return true;
}

bool Population::print_to_file_by_species(char *filename) {
  std::ofstream outFile(filename,std::ios::out);
  //Make sure it worked
  if (!outFile) {
    std::cerr<<"Can't open "<<filename<<" for output"<<std::endl;
    return false;
  }

  bool result = print_to_file_by_species(outFile);

  outFile.close();

  return result;
}


bool Population::print_to_file_by_species(std::ostream& outFile) {
    for(auto &s: species)
        s->print_to_file(outFile);

	return true;
}

bool Population::epoch(int generation) {
#ifndef NDEBUG
    for(Organism &org: orgs.curr()) {
        assert(org.generation == generation - 1 );
    }
#endif

	real_t total=0.0; //Used to compute average fitness over all Organisms
	real_t overall_average;  //The average modified fitness among ALL organisms

	//The fractional parts of expected offspring that can be 
	//Used only when they accumulate above 1 for the purposes of counting
	//Offspring
	real_t skim; 
	int total_expected;  //precision checking
	int total_organisms = size(); // todo: get rid of this variable
    assert(total_organisms == NEAT::pop_size);
	int max_expected;
	Species *best_species = nullptr;
	int final_expected;

	std::vector<Species*> sorted_species;  //Species sorted by max fit org in Species
	int half_pop;

	//We can try to keep the number of species constant at this number
	int num_species=species.size();

	//Stick the Species pointers into a new Species list for sorting
	for(Species *s: species) {
		sorted_species.push_back(s);
	}

	//Sort the Species by max fitness (Use an extra list to do this)
	//These need to use ORIGINAL fitness
	//sorted_species.qsort(order_species);
    std::sort(sorted_species.begin(), sorted_species.end(), order_species);

	//Flag the lowest performing species over age 20 every 30 generations 
	//NOTE: THIS IS FOR COMPETITIVE COEVOLUTION STAGNATION DETECTION
    if(generation % 30 == 0) {
        for(size_t i = sorted_species.size(); i > 0; i--) {
            Species *s = sorted_species[i - 1];
            if(s->age >= 20) {
                s->obliterate = true;
                break;
            }
        }
    }

	std::cout<<"Number of Species: "<<num_species<<std::endl;

	//Use Species' ages to modify the objective fitness of organisms
	// in other words, make it more fair for younger species
	// so they have a chance to take hold
	//Also penalize stagnant species
	//Then adjust the fitness using the species size to "share" fitness
	//within a species.
	//Then, within each Species, mark for death 
	//those below survival_thresh*average
    for(Species *s: species) {
        s->adjust_fitness();
    }

	//Go through the organisms and add up their fitnesses to compute the
	//overall average
    for(Organism &o: orgs.curr()) {
        total += o.fitness;
    }
	overall_average=total/total_organisms;
	std::cout<<"Generation "<<generation<<": "<<"overall_average = "<<overall_average<<std::endl;

	//Now compute expected number of offspring for each individual organism
    for(Organism &o: orgs.curr()) {
		o.expected_offspring = o.fitness / overall_average;
	}

	//Now add those offspring up within each Species to get the number of
	//offspring per Species
	skim=0.0;
	total_expected=0;
    for(Species *s: species) {
		skim = s->count_offspring(skim);
		total_expected += s->expected_offspring;
	}    

	//Need to make up for lost foating point precision in offspring assignment
	//If we lost precision, give an extra baby to the best Species
	if (total_expected<total_organisms) {
		//Find the Species expecting the most
		max_expected=0;
		final_expected=0;
		for(Species *s: species) {
			if (s->expected_offspring >= max_expected) {
				max_expected = s->expected_offspring;
				best_species = s;
			}
			final_expected += s->expected_offspring;
		}
		//Give the extra offspring to the best species
		++(best_species->expected_offspring);
		final_expected++;

		//If we still arent at total, there is a problem
		//Note that this can happen if a stagnant Species
		//dominates the population and then gets killed off by its age
		//Then the whole population plummets in fitness
		//If the average fitness is allowed to hit 0, then we no longer have 
		//an average we can use to assign offspring.
		if (final_expected < total_organisms) {
            for(Species *s: species) {
                s->expected_offspring = 0;
            }
			best_species->expected_offspring = total_organisms;
		}
	}

	//Sort the Species by max fitness (Use an extra list to do this)
	//These need to use ORIGINAL fitness
	//sorted_species.qsort(order_species);
    std::sort(sorted_species.begin(), sorted_species.end(), order_species);

	//Check for Population-level stagnation
    {
        Organism *pop_champ = sorted_species[0]->first();
        if(pop_champ->orig_fitness > highest_fitness) {
            real_t old_highest = highest_fitness;
            highest_fitness = pop_champ->orig_fitness;
            highest_last_changed=0;

            printf("NEW POPULATION RECORD FITNESS: %lg, delta=%lg @ gen=%d\n",
                   highest_fitness, highest_fitness - old_highest, generation);
        } else {
            ++highest_last_changed;

            printf("%zu generations since last population fitness record: %lg\n",
                   size_t(highest_last_changed), highest_fitness);
        }
    }

	//Check for stagnation- if there is stagnation, perform delta-coding
	if (highest_last_changed >= NEAT::dropoff_age+5) {
		highest_last_changed = 0;
		half_pop = total_organisms / 2;

		sorted_species[0]->first()->super_champ_offspring = half_pop;
		sorted_species[0]->expected_offspring = half_pop;
		sorted_species[0]->age_of_last_improvement = sorted_species[0]->age;

        if(sorted_species.size() > 1) {
            sorted_species[1]->first()->super_champ_offspring = total_organisms - half_pop;
			sorted_species[1]->expected_offspring = total_organisms - half_pop;
			sorted_species[1]->age_of_last_improvement = sorted_species[1]->age;

			//Get rid of all species under the first 2
            for(size_t i = 2, n = sorted_species.size(); i < n; i++) {
                sorted_species[i]->expected_offspring = 0;
			}
		} else {
            sorted_species[0]->first()->super_champ_offspring += total_organisms - half_pop;
            sorted_species[0]->expected_offspring += total_organisms - half_pop;
		}
	} else if (NEAT::babies_stolen>0) {
        //todo: catch at ne parsing
        trap("stolen babies no longer supported!");
	}


	//Kill off all Organisms marked for death.  The remainder
	//will be allowed to reproduce.
    for(Species *s: species) {
        s->remove_eliminated();
    }

    assert(total_expected <= (int)size());

    //Create the next generation.
    {
        static Timer timer("reproduce");
        timer.start();

        orgs.swap();

        //Initialize the parms for each reproduce invocation
        struct reproduce_parms_t {
            Species *species;
            int ioffspring;
        } reproduce_parms[size()];
        {
            size_t iorg = 0;
            for(size_t i = 0, n = species.size(); i < n; i++) {
                Species *s = species[i];

                for(int j = 0; j < s->expected_offspring; j++) {
                    reproduce_parms[iorg].species = s;
                    reproduce_parms[iorg].ioffspring = j;
                    iorg++;
                }
            }
            assert(iorg == size());
        }

#pragma omp parallel for
        for(size_t iorg = 0; iorg < size(); iorg++) {
            Organism &baby = orgs.curr()[iorg];
            baby.init(0.0, generation);
            baby.genome.reset(iorg+1);
            
            reproduce_parms_t &parms = reproduce_parms[iorg];

            parms.species->reproduce(iorg,
                                     parms.ioffspring,
                                     baby,
                                     this,
                                     sorted_species);
        }

        innovations.apply();

        //Create the neural nets for the new organisms.
        for(Organism &baby: orgs.curr())
            baby.create_phenotype();

        timer.stop();
    }

    {
        static Timer timer("speciate");
        timer.start();

        {
            size_t n = size();
#pragma omp parallel for
            for(size_t i = 0; i < n; i++) {
                Organism &org = orgs.curr()[i];
                org.species = nullptr;

                for(Species *s: species) {
                    if(s->size()) {
                        real_t comp = org.genome.compatibility(&s->first()->genome);
                        if(comp < NEAT::compat_threshold) {
                            org.species = s;
                            break;
                        }
                    }
                }
            }
        }

        size_t index_new_species = species.size();

        for(Organism &org: orgs.curr()) {
            if(!org.species) {
                //It didn't fit into any of the existing species. Check if it fits
                //into one we've just created.
                for(size_t i = index_new_species, n = species.size();
                    i < n;
                    i++) {

                    Species *s = species[i];
                    real_t comp = org.genome.compatibility(&s->first()->genome);
                    if(comp < NEAT::compat_threshold) {
                        org.species = s;
                        break;
                    }
                }
                //It didn't fit into a newly created species, so make one for it. 
                if(!org.species) {
                    org.species = new Species(++last_species, true);
                    species.push_back(org.species);
                }
            }
            org.species->add_Organism(&org);
        }

        timer.stop();
    }

	//Destroy and remove the old generation from the organisms and species
    for(Species *s: species) {
        s->remove_generation(generation - 1);
    }

	//Remove all empty Species and age ones that survive
	//As this happens, create master organism list for the new generation
    {
        size_t nspecies = 0;
        int orgcount = 0;

        for(size_t i = 0; i < species.size(); i++) {
            Species *s = species[i];
            if(s->organisms.empty()) {
                delete s;
            } else {
                species[nspecies++] = s;

                //Age surviving Species and 
                //Rebuild master Organism list: NUMBER THEM as they are added to the list
                if(s->novel) {
                    s->novel = false;
                } else {
                    s->age++;
                }
                
                //Go through the organisms of the curspecies and add them to 
                //the master list
                for(Organism *org: s->organisms) {
                    org->genome.genome_id = orgcount++;
                }
            }
        }

        species.resize(nspecies);
    }

#ifndef NDEBUG
    for(Organism &org: orgs.curr()) {
        assert(org.generation == generation);
    }
#endif

    {
        size_t nnodes = 0;
        size_t nlinks = 0;
        size_t ndisabled = 0;

        for(Organism &org: orgs.curr()) {
            nnodes += org.genome.nodes.size();
            nlinks += org.genome.links.size();
            for(LinkGene &g: org.genome.links)
                if(!g.enable)
                    ndisabled++;
        }

        real_t n = real_t(size());
        std::cout << "nnodes=" << (nnodes/n) << ", nlinks=" << (nlinks/n) << ", disabled=" << (ndisabled/real_t(nlinks)) << std::endl;
    }

	return true;
}

bool Population::rank_within_species() {
	std::vector<Species*>::iterator curspecies;

	//Add each Species in this generation to the snapshot
	for(curspecies=species.begin();curspecies!=species.end();++curspecies) {
		(*curspecies)->rank();
	}

	return true;
}
