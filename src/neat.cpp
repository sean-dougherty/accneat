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
#include "std.h" // Must be included first. Precompiled header with standard library includes.
#include "neat.h"

using NEAT::real_t;

const std::vector<NEAT::nodetype> NEAT::nodetypes = {
    NEAT::nodetype::BIAS,
    NEAT::nodetype::SENSOR,
    NEAT::nodetype::OUTPUT,
    NEAT::nodetype::HIDDEN
};

NEAT::NeatEnv *NEAT::env = new NeatEnv();

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



