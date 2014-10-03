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
#include "neat.h"

#include <fstream>
#include <cmath>
#include <cstring>

using NEAT::real_t;

const std::vector<NEAT::nodetype> NEAT::nodetypes = {
    NEAT::nodetype::BIAS,
    NEAT::nodetype::SENSOR,
    NEAT::nodetype::OUTPUT,
    NEAT::nodetype::HIDDEN
};

NEAT::GeneticSearchType NEAT::search_type = GeneticSearchType::COMPLEXIFY;
NEAT::PopulationType NEAT::population_type = PopulationType::SPECIES;
NEAT::GenomeType NEAT::genome_type = GenomeType::INNOV;
real_t NEAT::trait_param_mut_prob = 0.5;
real_t NEAT::trait_mutation_power = 1.0; // Power of mutation on a signle trait param 
real_t NEAT::linktrait_mut_sig = 1.0; // Amount that mutation_num changes for a trait change inside a link
real_t NEAT::nodetrait_mut_sig = 0.5; // Amount a mutation_num changes on a link connecting a node that changed its trait 
real_t NEAT::weight_mut_power = 1.8; // The power of a linkweight mutation 
real_t NEAT::recur_prob = 0.05; // Prob. that a link mutation which doesn't have to be recurrent will be made recurrent 
real_t NEAT::disjoint_coeff = 1.0;
real_t NEAT::excess_coeff = 1.0;
real_t NEAT::mutdiff_coeff = 3.0;
real_t NEAT::compat_threshold = 4.0;
real_t NEAT::age_significance = 1.0; // How much does age matter? 
real_t NEAT::survival_thresh = 0.4; // Percent of ave fitness for survival 
real_t NEAT::mutate_only_prob = 0.25; // Prob. of a non-mating reproduction 
real_t NEAT::mutate_random_trait_prob = 0.1;
real_t NEAT::mutate_link_trait_prob = 0.1;
real_t NEAT::mutate_node_trait_prob = 0.1;
real_t NEAT::mutate_link_weights_prob = 0.8;
real_t NEAT::mutate_toggle_enable_prob = 0.1;
real_t NEAT::mutate_gene_reenable_prob = 0.05;
real_t NEAT::mutate_add_node_prob = 0.01;
real_t NEAT::mutate_delete_node_prob = 0.01;
real_t NEAT::mutate_add_link_prob = 0.3;
real_t NEAT::mutate_delete_link_prob = 0.3;
bool NEAT::mutate_add_link_reenables = false;
real_t NEAT::interspecies_mate_rate = 0.001; // Prob. of a mate being outside species 
real_t NEAT::mate_multipoint_prob = 0.6;     
real_t NEAT::mate_only_prob = 0.2; // Prob. of mating without mutation 
real_t NEAT::recur_only_prob = 0.2;  // Probability of forcing selection of ONLY links that are naturally recurrent 
int NEAT::pop_size = 1000;  // Size of population 
size_t NEAT::deme_count = 10;
int NEAT::dropoff_age = 15;  // Age where Species starts to be penalized 
int NEAT::newlink_tries = 20;  // Number of tries mutate_add_link will attempt to find an open link 
int NEAT::print_every = 1000; // Tells to print population to file every n generations 
int NEAT::num_runs = 1;

int NEAT::getUnitCount(const char *string, const char *set)
{
	int count = 0;
	short last = 0;
	while(*string)
	{
		last = *string++;

		for(int i =0; set[i]; i++)
		{
			if(last == set[i])
			{
				count++;
				last = 0;
				break;
			}   
		}
	}
	if(last)
		count++;
	return count;
}   

real_t NEAT::oldhebbian(real_t weight, real_t maxweight, real_t active_in, real_t active_out, real_t hebb_rate, real_t pre_rate, real_t post_rate) {

	bool neg=false;
	real_t delta;

	//real_t weight_mag;

	if (maxweight<5.0) maxweight=5.0;

	if (weight>maxweight) weight=maxweight;

	if (weight<-maxweight) weight=-maxweight;

	if (weight<0) {
		neg=true;
		weight=-weight;
	}

	if (!(neg)) {
		//if (true) {
		delta=
			hebb_rate*(maxweight-weight)*active_in*active_out+
			pre_rate*(weight)*active_in*(active_out-1.0)+
			post_rate*(weight)*(active_in-1.0)*active_out;

		if (weight+delta>0)
			return weight+delta;
	}
	else {
		//In the inhibatory case, we strengthen the synapse when output is low and
		//input is high
		delta=
			hebb_rate*(maxweight-weight)*active_in*(1.0-active_out)+ //"unhebb"
			//hebb_rate*(maxweight-weight)*(1.0-active_in)*(active_out)+
			-5*hebb_rate*(weight)*active_in*active_out+ //anti-hebbian
			//hebb_rate*(maxweight-weight)*active_in*active_out+
			//pre_rate*weight*active_in*(active_out-1.0)+
			//post_rate*weight*(active_in-1.0)*active_out;
			0;

		//delta=delta-hebb_rate; //decay

		if (-(weight+delta)<0)
			return -(weight+delta);
		else return -0.01;

		return -(weight+delta);

	}

	return 0;

}

real_t NEAT::hebbian(real_t weight, real_t maxweight, real_t active_in, real_t active_out, real_t hebb_rate, real_t pre_rate, real_t post_rate) {

	bool neg=false;
	real_t delta;

	//real_t weight_mag;

	real_t topweight;

	if (maxweight<5.0) maxweight=5.0;

	if (weight>maxweight) weight=maxweight;

	if (weight<-maxweight) weight=-maxweight;

	if (weight<0) {
		neg=true;
		weight=-weight;
	}


	//if (weight<0) {
	//  weight_mag=-weight;
	//}
	//else weight_mag=weight;


	topweight=weight+2.0;
	if (topweight>maxweight) topweight=maxweight;

	if (!(neg)) {
		//if (true) {
		delta=
			hebb_rate*(maxweight-weight)*active_in*active_out+
			pre_rate*(topweight)*active_in*(active_out-1.0);
		//post_rate*(weight+1.0)*(active_in-1.0)*active_out;

		return weight+delta;

	}
	else {
		//In the inhibatory case, we strengthen the synapse when output is low and
		//input is high
		delta=
			pre_rate*(maxweight-weight)*active_in*(1.0-active_out)+ //"unhebb"
			//hebb_rate*(maxweight-weight)*(1.0-active_in)*(active_out)+
			-hebb_rate*(topweight+2.0)*active_in*active_out+ //anti-hebbian
			//hebb_rate*(maxweight-weight)*active_in*active_out+
			//pre_rate*weight*active_in*(active_out-1.0)+
			//post_rate*weight*(active_in-1.0)*active_out;
			0;

		//delta=delta-hebb_rate; //decay

		return -(weight+delta);
	}

}



