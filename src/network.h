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
#ifndef _NETWORK_H_
#define _NETWORK_H_

#include <algorithm>
#include <vector>
#include "neat.h"
#include "nnode.h"

namespace NEAT {

	// ----------------------------------------------------------------------- 
	// A NETWORK is a LIST of input NODEs and a LIST of output NODEs           
	//   The point of the network is to define a single entity which can evolve
	//   or learn on its own, even though it may be part of a larger framework 
	class Network {
    private:
        friend class Genome; // todo: remove friend. just for testing genome topology

		std::vector<NNode*> all_nodes;  // A list of all the nodes
		std::vector<NNode*> inputs;  // NNodes that input into the network
		std::vector<NNode*> outputs; // Values output by the network
		int net_id; // Allow for a network id
		bool adaptable; // Tells whether network can adapt or not
		double maxweight; // Maximum weight in network for adaptation purposes

	public:
		Network(std::vector<NNode*> in,
                std::vector<NNode*> out,
                std::vector<NNode*> all,
                int netid, 
                bool adaptval,
                double maxweight);

		// Copy Constructor
		Network(const Network& network);

		~Network();

		// Puts the network back into an inactive state
		void flush();
		
		// Activates the net such that all outputs are active
		bool activate();

		// Prints the values of its outputs
		void show_activation();

		// Takes an array of sensor values and loads it into SENSOR inputs ONLY
		void load_sensors(const double*);
		void load_sensors(const std::vector<double> &sensvals);

        double get_output(size_t index);

		// This checks a POTENTIAL link between a potential in_node
		// and potential out_node to see if it must be recurrent 
		// Use count and thresh to jump out in the case of an infinite loop 
		bool is_recur(NNode *potin_node,NNode *potout_node,int &count,int thresh); 

		// If all output are not active then return true
		bool outputsoff();

		// Just print connections weights with carriage returns
		void print_links_tofile(char *filename);

		int max_depth();

	};

} // namespace NEAT

#endif
