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
    public: // todo: remove public
        friend class Organism; // todo: remove friend

		std::vector<NNode> nodes;  // A list of all the nodes
        size_t ninput_nodes;
        size_t noutput_nodes;

		bool adaptable; // Tells whether network can adapt or not
		real_t maxweight; // Maximum weight in network for adaptation purposes
    public:
        Network();
		~Network();

        void reset();
        void init(real_t maxweight);

		// Puts the network back into an inactive state
		void flush();
		
		// Activates the net such that all outputs are active
		void activate();

		// Prints the values of its outputs
		void show_activation();

		// Takes an array of sensor values and loads it into SENSOR inputs ONLY
		void load_sensors(const real_t*);
		void load_sensors(const std::vector<real_t> &sensvals);

        real_t get_output(size_t index);
	};

} // namespace NEAT

#endif
