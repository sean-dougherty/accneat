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
#include "network.h"

#include <assert.h>

#include <iostream>
#include <sstream>

using namespace NEAT;
using std::cerr;
using std::endl;

Network::Network(std::vector<NNode> &&nodes_,
                 int netid,
                 bool adaptval,
                 double maxweight_)
    : nodes(std::move(nodes_)) {

    size_t i = 0;
    for(i = 0; (i < nodes.size()) && (nodes[i].type == SENSOR); i++) {
    }
    ninput_nodes = i;
    assert(ninput_nodes > 0);

    for(; (i < nodes.size()) && (nodes[i].type == NEURON) && (nodes[i].gen_node_label == OUTPUT); i++) {
    }

    noutput_nodes = i - ninput_nodes;
    assert(noutput_nodes > 0);

    for(; (i < nodes.size()); i++) {
        if(nodes[i].type != NEURON) {
            cerr << "Bad neuron type at " << i << ": " << nodes[i].type << endl;
            exit(1);
        }
        if(nodes[i].gen_node_label != HIDDEN) {
            cerr << "Bad neuron 'place' at " << i << ": " << nodes[i].gen_node_label << endl;
            exit(1);
        }
    }

    net_id = netid;
    adaptable = adaptval;
    maxweight = maxweight_;
}

Network::~Network() {
}

// Puts the network back into an initial state
void Network::flush() {
    for(size_t i = 0; i < noutput_nodes; i++) {
        NNode &node = nodes[i + ninput_nodes];
        assert(node.gen_node_label == OUTPUT);
        node.flushback(nodes);
    }
}

// If all output are not active then return true
bool Network::outputsoff() {
    for(size_t i = 0; i < noutput_nodes; i++) {
        NNode &node = nodes[i + ninput_nodes];
        assert(node.gen_node_label == OUTPUT);
        if(node.activation_count == 0)
            return true;
    }
    return false;
}

// Activates the net such that all outputs are active
// Returns true on success;
bool Network::activate() {
	double add_amount;  //For adding to the activesum
	bool onetime; //Make sure we at least activate once
	int abortcount=0;  //Used in case the output is somehow truncated from the network

	//Keep activating until all the outputs have become active 
	//(This only happens on the first activation, because after that they
	// are always active)

	onetime=false;

	while(outputsoff()||!onetime) {

		if(++abortcount==20) {
			return false;
		}

        // For each non-sensor node, compute the sum of its incoming activation
        for(size_t i = ninput_nodes; i < nodes.size(); i++) {
            NNode &node = nodes[i];
            node.activesum = 0;
            node.active_flag = false;  //This will tell us if it has any active inputs

            // For each incoming connection, add the activity from the connection to the activesum 
            for(Link &link: node.incoming) {
                NNode &inode = nodes[link.in_node_index];

                //Handle possible time delays
                if (!(link.time_delay)) {
                    add_amount=(link.weight)*(inode.get_active_out());
                    if ((inode.active_flag)||
                        (inode.type==SENSOR)) node.active_flag=true;
                    node.activesum+=add_amount;
                } else {
                    //Input over a time delayed connection
                    add_amount=(link.weight)*(inode.get_active_out_td());
                    node.activesum+=add_amount;
                }

            } //End for over incoming links
        }

		// Now activate all the non-sensor nodes off their incoming activation 
        for(size_t i = ninput_nodes; i < nodes.size(); i++) {
            NNode &node = nodes[i];
            //Only activate if some active input came in
            if (node.active_flag) {
                //Keep a memory of activations for potential time delayed connections
                node.last_activation2=node.last_activation;
                node.last_activation=node.activation;
                //Now run the net activation through an activation function
                node.activation=NEAT::fsigmoid(node.activesum,4.924273,2.4621365);  //Sigmoidal activation- see comments under fsigmoid
                //Increment the activation_count
                //First activation cannot be from nothing!!
                node.activation_count++;
            }
		}

		onetime=true;
	}

	if (adaptable) {
        // ADAPTATION:  Adapt weights based on activations 
        for(size_t i = ninput_nodes; i < nodes.size(); i++) {
            // For each incoming connection, perform adaptation based on the trait of the connection 
            NNode &node = nodes[i];
            for(Link &link: node.incoming) {
                NNode &inode = nodes[link.in_node_index];
                NNode &onode = nodes[link.out_node_index];
		
                if ((link.trait_id==2)||
                    (link.trait_id==3)||
                    (link.trait_id==4)) {
		  
                    //In the recurrent case we must take the last activation of the input for calculating hebbian changes
                    if (link.is_recurrent) {
                        link.weight=
                            hebbian(link.weight,maxweight,
                                    inode.last_activation, 
                                    onode.get_active_out(),
                                    link.params[0],link.params[1],
                                    link.params[2]);
		    
		    
                    }
                    else { //non-recurrent case
                        link.weight=
                            hebbian(link.weight,maxweight,
                                    inode.get_active_out(), 
                                    onode.get_active_out(),
                                    link.params[0],link.params[1],
                                    link.params[2]);
                    }
                }
		
            }	      
        }
	} //end if (adaptable)

	return true;  
}

// Takes an array of sensor values and loads it into SENSOR inputs ONLY
void Network::load_sensors(const double *sensvals) {
    for(size_t i = 0; i < ninput_nodes; i++) {
        nodes[i].sensor_load(sensvals[i]);
    }
}

void Network::load_sensors(const std::vector<double> &sensvals) {
    assert(sensvals.size() == ninput_nodes);

    load_sensors(sensvals.data());
}

double Network::get_output(size_t index) {
    assert(index < noutput_nodes);

    return nodes[ninput_nodes + index].activation;
}

//Find the maximum number of neurons between an ouput and an input
int Network::max_depth() {
    int cur_depth; //The depth of the current node
    int max = 0; //The max depth

    for(size_t i = 0; i < noutput_nodes; i++) {
        cur_depth = nodes[i + ninput_nodes].depth(0, nodes);
        if(cur_depth > max) max = cur_depth;
    }

    return max;
}

