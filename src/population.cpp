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
#include <assert.h>
#include <iostream>
#include <sstream>
#include <fstream>

using namespace NEAT;
using std::vector;

Population::Population(Genome *g,int size) {
	winnergen=0;
	highest_fitness=0.0;
	highest_last_changed=0;
	spawn(g,size);
}

Population::Population(Genome *g,int size, float power) {
	winnergen=0;
	highest_fitness=0.0;
	highest_last_changed=0;
	clone(g, size, power);
}

Population::Population(const char *filename) {

	char curword[128];  //max word size of 128 characters
	char curline[1024]; //max line size of 1024 characters

	Genome *new_genome;

	winnergen=0;

	highest_fitness=0.0;
	highest_last_changed=0;

	cur_node_id=0;
	cur_innov_num=0.0;

	std::ifstream iFile(filename);
	if (!iFile) {
		printf("Can't open genomes file for input");
		return;
	}

	else {
		bool md = false;
		char metadata[128];
		//Loop until file is finished, parsing each line
		while (!iFile.eof()) 
		{
			iFile.getline(curline, sizeof(curline));
            std::stringstream ss(curline);
            ss >> curword;

			//Check for next
			if (strcmp(curword,"genomestart")==0) 
			{
                int idcheck;
                ss >> idcheck;

				// If there isn't metadata, set metadata to ""
				if(md == false)  {
					strcpy(metadata, "");
				}
				md = false;

                Organism *org = new Organism();
                org->init(0, 1, metadata);
				org->genome.load_from_file(idcheck, iFile);
                org->create_phenotype();
				organisms.push_back(org);
				if (cur_node_id<(new_genome->get_last_node_id()))
					cur_node_id=new_genome->get_last_node_id();

				if (cur_innov_num<(new_genome->get_last_gene_innovnum()))
					cur_innov_num=new_genome->get_last_gene_innovnum();
			}
			else if (strcmp(curword,"/*")==0) 
			{
				// New metadata possibly, so clear out the metadata
				strcpy(metadata, "");
                ss >> curword;

				while(strcmp(curword,"*/")!=0)
				{
					// If we've started to form the metadata, put a space in the front
					if(md) {
						strncat(metadata, " ", 128 - strlen(metadata));
					}

					// Append the next word to the metadata, and say that there is metadata
					strncat(metadata, curword, 128 - strlen(metadata));
					md = true;

					//strcpy(curword, NEAT::getUnit(curline, curwordnum++, delimiters));
                    ss >> curword;
				}
			}
			//Ignore comments - they get printed to screen
			//else if (strcmp(curword,"/*")==0) {
			//	iFile>>curword;
			//	while (strcmp(curword,"*/")!=0) {
			//cout<<curword<<" ";
			//		iFile>>curword;
			//	}
			//	cout<<endl;

			//}
			//Ignore comments surrounded by - they get printed to screen
		}

		iFile.close();

		speciate();

	}
}


Population::~Population() {

	std::vector<Species*>::iterator curspec;
	std::vector<Organism*>::iterator curorg;
	//std::vector<Generation_viz*>::iterator cursnap;

	if (species.begin()!=species.end()) {
		for(curspec=species.begin();curspec!=species.end();++curspec) {
			delete (*curspec);
		}
	}
	else {
		for(curorg=organisms.begin();curorg!=organisms.end();++curorg) {
			delete (*curorg);
		}
	}

	for (std::vector<Innovation*>::iterator iter = innovations.begin(); iter != innovations.end(); ++iter)
		delete *iter;

	//Delete the snapshots
	//		for(cursnap=generation_snapshots.begin();cursnap!=generation_snapshots.end();++cursnap) {
	//			delete (*cursnap);
	//		}
}

void Population::verify() {
    for(auto &org: organisms)
        org->genome.verify();
} 

bool Population::clone(Genome *g,int size, float power) {
	int count;
	Genome *new_genome;
	Organism *new_organism;

    new_organism = new Organism();
    new_genome = &new_organism->genome;
    g->duplicate_into(*new_genome, 1);
    new_organism->create_phenotype();
	organisms.push_back(new_organism);
	
	//Create size copies of the Genome
	//Start with perturbed linkweights
	for(count=2;count<=size;count++) {
		new_organism = new Organism();
        new_genome = &new_organism->genome;
		g->duplicate_into(*new_genome, count); 
		if(power>0)
			new_genome->mutate_link_weights(power,1.0,GAUSSIAN);
		
		new_genome->randomize_traits();
        new_organism->create_phenotype();
		organisms.push_back(new_organism);
	}

	//Keep a record of the innovation and node number we are on
	cur_node_id=new_genome->get_last_node_id();
	cur_innov_num=new_genome->get_last_gene_innovnum();

	//Separate the new Population into species
	speciate();

	return true;
}

bool Population::spawn(Genome *g,int size) {
	int count;
	Genome *new_genome;
	Organism *new_organism;

	//Create size copies of the Genome
	//Start with perturbed linkweights
	for(count=1;count<=size;count++) {
		new_organism = new Organism();
		new_genome = &new_organism->genome;
        g->duplicate_into(*new_genome, count); 
		new_genome->mutate_link_weights(1.0,1.0,COLDGAUSSIAN);
		new_genome->randomize_traits();
        new_organism->create_phenotype();
		organisms.push_back(new_organism);
	}

	//Keep a record of the innovation and node number we are on
	cur_node_id=new_genome->get_last_node_id();
	cur_innov_num=new_genome->get_last_gene_innovnum();

	//Separate the new Population into species
	speciate();

	return true;
}

bool Population::speciate() {
    last_species = 0;
    for(Organism *org: organisms) {
        assert(org->species == nullptr);
        for(Species *s: species) {
            if( org->genome.compatibility(&s->first()->genome) < NEAT::compat_threshold ) {
                org->species = s;
                break;
            }
        }
        if(!org->species) {
            Species *s = new Species(++last_species);
            species.push_back(s);
            org->species = s;
        }
        org->species->add_Organism(org);
    }
    return true;
}

bool Population::print_to_file_by_species(char *filename) {

  std::vector<Species*>::iterator curspecies;

  std::ofstream outFile(filename,std::ios::out);

  //Make sure it worked
  if (!outFile) {
    std::cerr<<"Can't open "<<filename<<" for output"<<std::endl;
    return false;
  }


  //Step through the Species and print them to the file
  for(curspecies=species.begin();curspecies!=species.end();++curspecies) {
    (*curspecies)->print_to_file(outFile);
  }

  outFile.close();

  return true;

}


bool Population::print_to_file_by_species(std::ostream& outFile) {
    for(auto &s: species)
        s->print_to_file(outFile);

	return true;
}

bool Population::epoch(int generation) {
    std::vector<Species *>::iterator curspecies;

	double total=0.0; //Used to compute average fitness over all Organisms

	double overall_average;  //The average modified fitness among ALL organisms

	//The fractional parts of expected offspring that can be 
	//Used only when they accumulate above 1 for the purposes of counting
	//Offspring
	double skim; 
	int total_expected;  //precision checking
	int total_organisms = organisms.size();
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
    for(Organism *o: organisms) {
        total += o->fitness;
    }
	overall_average=total/total_organisms;
	std::cout<<"Generation "<<generation<<": "<<"overall_average = "<<overall_average<<std::endl;

	//Now compute expected number of offspring for each individual organism
    for(Organism *o: organisms) {
		o->expected_offspring = o->fitness / overall_average;
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
        pop_champ->pop_champ = true; //DEBUG marker of the best of pop
        if(pop_champ->orig_fitness > highest_fitness) {
            double old_highest = highest_fitness;
            highest_fitness = pop_champ->orig_fitness;
            highest_last_changed=0;

            printf("NEW POPULATION RECORD FITNESS: %lg, delta=%lg\n",
                   highest_fitness, highest_fitness - old_highest);
        } else {
            ++highest_last_changed;

            printf("%uz generations since last population fitness record: %lg\n",
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
    {
        for(Species *s: species) {
            s->remove_eliminated();
        }

        size_t n = 0;
        for(size_t i = 0; i < organisms.size(); i++) {
            Organism *org = organisms[i];
            if(org->eliminate) {
                delete org;
            } else {
                organisms[n++] = organisms[i];
            }
        }
        organisms.resize(n);
    }

    //Perform reproduction within the species. Note that new species may
    //be created as we iterate over the vector.
    for(size_t i = 0, n = species.size(); i < n; i++) {
        vector<Organism *> offspring = species[i]->reproduce(generation,
                                                             this,
                                                             sorted_species);

        for(Organism *org: offspring) {
            assert(org->species == nullptr);

            for(Species *s: species) {
                if(s->size()) {
                    double comp = org->genome.compatibility(&s->first()->genome);
                    if(comp < NEAT::compat_threshold) {
                        org->species = s;
                        break;
                    }
                }
            }

            if(!org->species) {
                org->species = new Species(++last_species, true);
                species.push_back(org->species);
            }
            org->species->add_Organism(org);
        }
    }

	//Destroy and remove the old generation from the organisms and species
    for(Organism *org: organisms) {
        org->species->remove_org(org);
        delete org;
    }
    organisms.clear();

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
                    organisms.push_back(org);
                }
            }
        }

        species.resize(nspecies);
    }

	//Remove the innovations of the current generation
    for(Innovation *innov: innovations) {
        delete innov;
    }
    innovations.clear();

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
