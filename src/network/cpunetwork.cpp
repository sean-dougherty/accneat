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

    activations.resize(nnodes);

    flush();
}

Network &CpuNetwork::operator=(const Network &other_) {
    const CpuNetwork &other = dynamic_cast<const CpuNetwork &>(other_);

    this->counts = other.counts;
    this->nodes = other.nodes;
    this->links = other.links;
    this->activations = other.activations;

    return *this;
}

// Puts the network into an initial state
void CpuNetwork::flush() {
    for(size_t i = 0; i < counts.nbias_nodes; i++) {
        activations[i] = 1.0;
    }
    for(size_t i = counts.nbias_nodes; i < counts.nnodes; i++) {
        activations[i] = 0.0;
    }
}

void CpuNetwork::activate(size_t ncycles) {
    real_t act_other[counts.nnodes];

    //Copy only input activation state
    memcpy(act_other, activations.data(), sizeof(real_t) * counts.ninput_nodes);

    real_t *act_curr = activations.data(), *act_new = act_other;

    for(size_t icycle = 0; icycle < ncycles; icycle++) {

        for(size_t i = counts.ninput_nodes; i < counts.nnodes; i++) {
            NetNode &node = nodes[i];

            real_t sum = 0.0;
            for(size_t j = node.incoming_start; j < node.incoming_end; j++) {
                NetLink &link = links[j];
                sum += link.weight * act_curr[link.in_node_index];
            }

            act_new[i] = NEAT::fsigmoid(sum,
                                        4.924273,
                                        2.4621365);  //Sigmoidal activation- see comments under fsigmoid
        }

        std::swap(act_curr, act_new);
    }

    if(act_curr != activations.data()) {
        // If an odd number of cycles, we have to copy non-input data
        // of act_other back into activations.
        memcpy(activations.data() + counts.ninput_nodes,
               act_other + counts.ninput_nodes,
               sizeof(real_t) * (counts.nnodes - counts.ninput_nodes));
    }
}

void CpuNetwork::load_sensors(const std::vector<real_t> &sensvals) {
    assert(sensvals.size() == counts.nsensor_nodes);

    for(size_t i = 0; i < counts.nsensor_nodes; i++) {
        activations[i + counts.nbias_nodes] = sensvals[i];
    }
}

real_t CpuNetwork::get_output(size_t index) {
    assert(index < counts.noutput_nodes);

    return activations[counts.ninput_nodes + index];
}
