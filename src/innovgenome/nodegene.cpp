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
#include "nodegene.h"
#include <iostream>
#include <sstream>
using namespace NEAT;

NodeGene::NodeGene(nodetype ntype,int nodeid, nodeplace placement) {
	type=ntype; //NEURON or SENSOR type
	node_id=nodeid;
	place=placement;
	frozen=false;
	trait_id=1;
}

NodeGene::NodeGene (const char *argline) {
    std::stringstream ss(argline);
    int nodety, nodepl;
    ss >> node_id >> trait_id >> nodety >> nodepl;
    type = (nodetype)nodety;
    place = (nodeplace)nodepl;

    if(trait_id == 0)
        trait_id = 1;

	// Get the Sensor Identifier and Parameter String
	// mySensor = SensorRegistry::getSensor(id, param);
	frozen=false;  //TODO: Maybe change
}

NodeGene::~NodeGene() {
}

void NodeGene::print_to_file(std::ostream &outFile) {
  outFile<<"node "<<node_id<<" ";
  outFile<<trait_id<<" ";
  outFile<<type<<" ";
  outFile<<place<<std::endl;
}
