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
#include "util.h"
#include <assert.h>

using namespace NEAT;
using std::cerr;
using std::endl;

Network::Network() {
}

// Requires nodes to be sorted by type: BIAS, SENSOR, OUTPUT, HIDDEN
void Network::init(const NodeCounts &counts_,
                   NNode *nodes_, size_t nnodes,
                   Link *links_, size_t nlinks) {

    counts = counts_;

    nodes.resize(nnodes);
    for(size_t i = 0; i < nnodes; i++) {
        nodes[i] = nodes_[i];
    }

    links.resize(nlinks);
    for(size_t i = 0; i < nlinks; i++) {
        links[i] = links_[i];
    }

    activations_buffers[0].resize(nnodes);
    activations_buffers[1].resize(nnodes);
    activations = activations_buffers[0].data();
    last_activations = activations_buffers[1].data();

    flush();
}

Network::~Network() {
}

// Puts the network into an initial state
void Network::flush() {
    for(size_t i = 0; i < counts.nbias_nodes; i++) {
        activations[i] = last_activations[i] = 1.0;
    }
    for(size_t i = counts.nbias_nodes; i < counts.nnodes; i++) {
        activations[i] = last_activations[i] = 0.0;
    }
}

// Activates the net such that all outputs are active
void Network::activate() {
    std::swap(activations, last_activations);

    // For each non-sensor node, compute the sum of its incoming activation
    for(size_t i = counts.ninput_nodes; i < counts.nnodes; i++) {
        NNode &node = nodes[i];

        real_t sum = 0.0;
        for(size_t j = node.incoming_start; j < node.incoming_end; j++) {
            Link &link = links[j];
            sum += link.weight * last_activations[link.in_node_index];
        } //End for over incoming links

        activations[i] = NEAT::fsigmoid(sum,
                                        4.924273,
                                        2.4621365);  //Sigmoidal activation- see comments under fsigmoid
    }
}

// Takes an array of sensor values and loads it into SENSOR inputs ONLY
void Network::load_sensors(const real_t *sensvals) {
    for(size_t i = 0; i < counts.nsensor_nodes; i++) {
        activations[i + counts.nbias_nodes] = last_activations[i + counts.nbias_nodes] = sensvals[i];
    }
}

void Network::load_sensors(const std::vector<real_t> &sensvals) {
    assert(sensvals.size() == counts.nsensor_nodes);

    load_sensors(sensvals.data());
}

real_t Network::get_output(size_t index) {
    assert(index < counts.noutput_nodes);

    return activations[counts.ninput_nodes + index];
}
