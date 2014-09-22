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

NNode::NNode(int node_id_,
             nodetype type_,
             nodeplace place_) {
	activation=0;
	last_activation=0;
    node_id = node_id_;
    type = type_;
    place = place_;
}

NNode::~NNode() {
}

void NNode::flush() {
    if(type != SENSOR) {
        activation = 0.0;
        last_activation = 0.0;
    }
}

// Sets activation level of sensor
void NNode::sensor_load(real_t value) {
    assert(type==SENSOR);

    last_activation = activation = value;
}
