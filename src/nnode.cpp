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
#include "nnode.h"
#include <iostream>
#include <sstream>
using namespace NEAT;
using std::vector;

NNode::NNode(nodetype ntype,int nodeid) {
    in_depth = false;
	active_flag=false;
	activesum=0;
	activation=0;
	last_activation=0;
	last_activation2=0;
	type=ntype; //NEURON or SENSOR type
	activation_count=0; //Inactive upon creation
	node_id=nodeid;
	ftype=SIGMOID;
	gen_node_label=HIDDEN;
	frozen=false;
	trait_id=1;
}

NNode::NNode(nodetype ntype,int nodeid, nodeplace placement) {
    in_depth = false;
	active_flag=false;
	activesum=0;
	activation=0;
	last_activation=0;
	last_activation2=0;
	type=ntype; //NEURON or SENSOR type
	activation_count=0; //Inactive upon creation
	node_id=nodeid;
	ftype=SIGMOID;
	gen_node_label=placement;
	frozen=false;
	trait_id=1;
}

NNode NNode::partial_copy(NNode *n) {
    NNode copy;
    copy.in_depth = false;
	copy.active_flag=false;
	copy.activation=0;
	copy.last_activation=0;
	copy.last_activation2=0;
	copy.type=n->type; //NEURON or SENSOR type
	copy.activation_count=0; //Inactive upon creation
	copy.node_id=n->node_id;
	copy.ftype=SIGMOID;
	copy.gen_node_label=n->gen_node_label;
	copy.frozen=false;
    copy.trait_id = n->trait_id;
    return copy;
}

NNode::NNode (const char *argline) {
    in_depth = false;

	activesum=0;

    std::stringstream ss(argline);
    int nodety, nodepl;
    ss >> node_id >> trait_id >> nodety >> nodepl;
    type = (nodetype)nodety;
    gen_node_label = (nodeplace)nodepl;

    if(trait_id == 0)
        trait_id = 1;

	// Get the Sensor Identifier and Parameter String
	// mySensor = SensorRegistry::getSensor(id, param);
	frozen=false;  //TODO: Maybe change
}

NNode::~NNode() {
}

//Returns the type of the node, NEURON or SENSOR
const nodetype NNode::get_type() {
	return type;
}

//Allows alteration between NEURON and SENSOR.  Returns its argument
nodetype NNode::set_type(nodetype newtype) {
	type=newtype;
	return newtype;
}

//If the node is a SENSOR, returns true and loads the value
bool NNode::sensor_load(double value) {
	if (type==SENSOR) {

		//Time delay memory
		last_activation2=last_activation;
		last_activation=activation;

		activation_count++;  //Puts sensor into next time-step
		activation=value;
		return true;
	}
	else return false;
}

// Return activation currently in node from PREVIOUS (time-delayed) time step,
// if there is one
double NNode::get_active_out_td() {
	if (activation_count>1)
		return last_activation;
	else return 0.0;
}

// This recursively flushes everything leading into and including this NNode, including recurrencies
//todo: this doesn't need to be recursive
void NNode::flushback(vector<NNode> &nodes) {
	//A sensor should not flush black
	if (type!=SENSOR) {

		if (activation_count>0) {
			activation_count=0;
			activation=0;
			last_activation=0;
			last_activation2=0;
		}

		//Flush back recursively
		for(Link &link: incoming) {
			//Flush the link itself (For future learning parameters possibility) 
			link.added_weight=0;
            NNode &inode = nodes[link.in_node_index];
			if(inode.activation_count > 0)
				inode.flushback(nodes);
		}
	} else {
		//Flush the SENSOR
		activation_count=0;
		activation=0;
		last_activation=0;
		last_activation2=0;

	}
}

// Reserved for future system expansion
void NNode::derive_trait(const Trait &t) {
    trait_id = t.trait_id;
    for(int count=0; count < NEAT::num_trait_params; count++)
        params[count] = t.params[count];
}

void NNode::print_to_file(std::ostream &outFile) {
  outFile<<"node "<<node_id<<" ";
  outFile<<trait_id<<" ";
  outFile<<type<<" ";
  outFile<<gen_node_label<<std::endl;
}

//Find the greatest depth starting from this neuron at depth d
int NNode::depth(int d, vector<NNode> &nodes) {
    const int MAX_DEPTH = 10;
    int cur_depth; //The depth of the current node
    int max=d; //The max depth

    if (d>=MAX_DEPTH) {
        return MAX_DEPTH;
    }

    //Base Case
    if ((this->type)==SENSOR) {
        return d;
        //Recursion
    } else {

        for(Link &link: incoming) {
            cur_depth = nodes[link.in_node_index].depth(d+1, nodes);
            if (cur_depth>max) max=cur_depth;
        }
  
        return max;

    } //end else

}
