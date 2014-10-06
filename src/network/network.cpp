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
#include "network.h"
#include <assert.h>

using namespace NEAT;
using std::cerr;
using std::endl;

Network::Network() {
}

void Network::reset() {
    nodes.clear();
    ninput_nodes = 0;
    noutput_nodes = 0;
}

// Requires nodes to be sorted by type: BIAS, SENSOR, OUTPUT, HIDDEN
void Network::init(real_t maxweight_) {
    maxweight = maxweight_;

    size_t i = 0;

    for(; (i < nodes.size()) && (nodes[i].type == nodetype::BIAS); i++) {
    }
    nbias_nodes = i;

    for(; (i < nodes.size()) && (nodes[i].type == nodetype::SENSOR); i++) {
    }
    ninput_nodes = i;
    nsensor_nodes = ninput_nodes - nbias_nodes;

    for(; (i < nodes.size()) && (nodes[i].type == nodetype::OUTPUT); i++) {
    }
    noutput_nodes = i - ninput_nodes;
    assert(noutput_nodes > 0);

    for(; (i < nodes.size()); i++) {
        if(nodes[i].type != nodetype::HIDDEN) {
            cerr << "Bad neuron type at " << i << ": " << (int)nodes[i].type << endl;
            abort();
        }
    }
}

Network::~Network() {
}

// Puts the network back into an initial state
void Network::flush() {
    for(NNode &node: nodes) {
        node.flush();
    }
}

// Activates the net such that all outputs are active
void Network::activate() {
    // For each non-sensor node, compute the sum of its incoming activation
    for(size_t i = ninput_nodes; i < nodes.size(); i++) {
        NNode &node = nodes[i];

        real_t activation = 0.0;
        // For each incoming connection, add the activity from the connection to the activesum 
        for(Link &link: node.incoming) {
            NNode &inode = nodes[link.in_node_index];

            activation += link.weight * inode.last_activation;
        } //End for over incoming links

        node.activation = NEAT::fsigmoid(activation,
                                         4.924273,
                                         2.4621365);  //Sigmoidal activation- see comments under fsigmoid
    }

    for(size_t i = ninput_nodes; i < nodes.size(); i++) {
        NNode &node = nodes[i];
        node.last_activation = node.activation;
    }
}

// Takes an array of sensor values and loads it into SENSOR inputs ONLY
void Network::load_sensors(const real_t *sensvals) {
    for(size_t i = 0; i < nsensor_nodes; i++) {
        nodes[i + nbias_nodes].sensor_load(sensvals[i]);
    }
}

void Network::load_sensors(const std::vector<real_t> &sensvals) {
    assert(sensvals.size() == nsensor_nodes);

    load_sensors(sensvals.data());
}

real_t Network::get_output(size_t index) {
    assert(index < noutput_nodes);

    return nodes[ninput_nodes + index].activation;
}
