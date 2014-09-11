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


Link::Link(int tid,
           double w,
           node_index_t inode_index,
           node_index_t onode_index,
           bool recur) {
    trait_id = tid;
	weight = w;
    in_node_index = inode_index;
    out_node_index = onode_index;
	is_recurrent = recur;
	added_weight = 0;
}	

Link::Link(double w,
           node_index_t inode_index,
           node_index_t onode_index,
           bool recur)
    : Link(1, w, inode_index, onode_index, recur) {
}

Link::Link(double w)
    : Link(1, w, 0, 0, false) {
}

void Link::derive_trait(const Trait &t) {
    trait_id = t.trait_id;
    for(int count=0; count < NEAT::num_trait_params; count++)
        params[count] = t.params[count];
}
