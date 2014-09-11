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
	activation=0;
	last_activation=0;
	type=ntype; //NEURON or SENSOR type
	node_id=nodeid;
	ftype=SIGMOID;
	gen_node_label=HIDDEN;
	frozen=false;
	trait_id=1;
}

NNode::NNode(nodetype ntype,int nodeid, nodeplace placement) {
	activation=0;
	last_activation=0;
	type=ntype; //NEURON or SENSOR type
	node_id=nodeid;
	ftype=SIGMOID;
	gen_node_label=placement;
	frozen=false;
	trait_id=1;
}

NNode NNode::partial_copy(NNode *n) {
    NNode copy;
	copy.activation=0;
	copy.last_activation=0;
	copy.type=n->type; //NEURON or SENSOR type
	copy.node_id=n->node_id;
	copy.ftype=SIGMOID;
	copy.gen_node_label=n->gen_node_label;
	copy.frozen=false;
    copy.trait_id = n->trait_id;
    return copy;
}

NNode::NNode (const char *argline) {
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

void NNode::flush() {
    if(type != SENSOR) {
        activation = 0.0;
        last_activation = 0.0;
    }
}

// Sets activation level of sensor
void NNode::sensor_load(double value) {
    assert(type==SENSOR);

    last_activation = activation = value;
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
