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
#include "cpunetwork.h"
#include "neat.h"
#include "util.h"
#include <assert.h>

using namespace NEAT;
using std::cerr;
using std::endl;

// Requires nodes to be sorted by type: BIAS, SENSOR, OUTPUT, HIDDEN
void CpuNetwork::configure(const NodeCounts &counts_,
                           NetNode *nodes_, node_size_t nnodes,
                           NetLink *links_, link_size_t nlinks) {
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

Network &CpuNetwork::operator=(const Network &other_) {
    const CpuNetwork &other = dynamic_cast<const CpuNetwork &>(other_);

    this->counts = other.counts;
    this->nodes = other.nodes;
    this->links = other.links;
    this->activations_buffers[0] = other.activations_buffers[0];
    this->activations_buffers[1] = other.activations_buffers[1];

    activations = activations_buffers[0].data();
    last_activations = activations_buffers[1].data();
    if(other.activations == other.activations_buffers[1].data()) {
        std::swap(activations, last_activations);
    }

    return *this;
}

// Puts the network into an initial state
void CpuNetwork::flush() {
    for(size_t i = 0; i < counts.nbias_nodes; i++) {
        activations[i] = last_activations[i] = 1.0;
    }
    for(size_t i = counts.nbias_nodes; i < counts.nnodes; i++) {
        activations[i] = last_activations[i] = 0.0;
    }
}

void CpuNetwork::activate(size_t ncycles) {
    for(size_t icycle = 0; icycle < ncycles; icycle++) {
        std::swap(activations, last_activations);

        // For each non-sensor node, compute the sum of its incoming activation
        for(size_t i = counts.ninput_nodes; i < counts.nnodes; i++) {
            NetNode &node = nodes[i];

            real_t sum = 0.0;
            for(size_t j = node.incoming_start; j < node.incoming_end; j++) {
                NetLink &link = links[j];
                sum += link.weight * last_activations[link.in_node_index];
            } //End for over incoming links

            activations[i] = NEAT::fsigmoid(sum,
                                            4.924273,
                                            2.4621365);  //Sigmoidal activation- see comments under fsigmoid
        }
    }
}

void CpuNetwork::load_sensors(const std::vector<real_t> &sensvals) {
    assert(sensvals.size() == counts.nsensor_nodes);

    for(size_t i = 0; i < counts.nsensor_nodes; i++) {
        activations[i + counts.nbias_nodes] = last_activations[i + counts.nbias_nodes] = sensvals[i];
    }
}

real_t CpuNetwork::get_output(size_t index) {
    assert(index < counts.noutput_nodes);

    return activations[counts.ninput_nodes + index];
}
