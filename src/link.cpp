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
#include "link.h"

using namespace NEAT;

Link::Link(double w,NNode *inode,NNode *onode,bool recur) {
	weight=w;
	in_node=inode;
	out_node=onode;
	is_recurrent=recur;
	added_weight=0;
	time_delay=false;
	trait_id=1;
}

Link::Link(int trait_id_,double w,NNode *inode,NNode *onode,bool recur) {
	weight=w;
	in_node=inode;
	out_node=onode;
	is_recurrent=recur;
	added_weight=0;
	time_delay=false;
    trait_id = trait_id_;
}	

Link::Link(double w) {
	weight=w;
	in_node=out_node=0;  
	is_recurrent=false;
	time_delay=false;
	trait_id=1;
}

Link::Link(const Link& link)
{
	weight = link.weight;
	in_node = link.in_node;
	out_node = link.out_node;
	is_recurrent = link.is_recurrent;
	added_weight = link.added_weight;
	time_delay = link.time_delay;
	trait_id = link.trait_id;
}

void Link::derive_trait(const Trait &t) {
    trait_id = t.trait_id;
    for(int count=0; count < NEAT::num_trait_params; count++)
        params[count] = t.params[count];
}
