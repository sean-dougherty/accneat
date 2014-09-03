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

Network::Network(std::vector<NNode*> in,
                 std::vector<NNode*> out,
                 std::vector<NNode*> all,
                 int netid,
                 bool adaptval,
                 double maxweight_) {
  inputs = in;
  outputs = out;
  all_nodes = all;
  net_id = netid;
  adaptable = adaptval;
  maxweight = maxweight_;
}


Network::Network(const Network& network)
{
	std::vector<NNode*>::const_iterator curnode;

	// Copy all the inputs
	for(curnode = network.inputs.begin(); curnode != network.inputs.end(); ++curnode) {
		NNode* n = new NNode(**curnode);
		inputs.push_back(n);
		all_nodes.push_back(n);
	}

	// Copy all the outputs
	for(curnode = network.outputs.begin(); curnode != network.outputs.end(); ++curnode) {
		NNode* n = new NNode(**curnode);
		outputs.push_back(n);
		all_nodes.push_back(n);
	}

	net_id = network.net_id;
	adaptable = network.adaptable;
}

Network::~Network() {
    for(NNode *node: all_nodes)
        delete node;
}

// Puts the network back into an initial state
void Network::flush() {
    for(NNode *node: outputs)
        node->flushback();
}

// If all output are not active then return true
bool Network::outputsoff() {
    for(NNode *node: outputs)
        if(node->activation_count == 0)
            return true;

    return false;
}

// Print the connections weights to a file separated by only carriage returns
void Network::print_links_tofile(char *filename) {
    std::ofstream oFile(filename);

    for(NNode* node: all_nodes) {
		if ((node->type)!=SENSOR) {
			for(Link &link: node->incoming) {
                oFile << link.in_node->node_id << " -> " << link.out_node->node_id << " : " << link.weight << std::endl;
			} // end for loop on links
		} //end if
	} //end for loop on nodes

	oFile.close();

} //print_links_tofile

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

		// For each node, compute the sum of its incoming activation 
		for(NNode *node: all_nodes) {
			if(node->type != SENSOR) {
				node->activesum=0;
				node->active_flag=false;  //This will tell us if it has any active inputs

				// For each incoming connection, add the activity from the connection to the activesum 
				for(Link &link: node->incoming) {
					//Handle possible time delays
					if (!(link.time_delay)) {
						add_amount=(link.weight)*((link.in_node)->get_active_out());
						if (((link.in_node)->active_flag)||
							((link.in_node)->type==SENSOR)) node->active_flag=true;
						node->activesum+=add_amount;
					}
					else {
						//Input over a time delayed connection
						add_amount=(link.weight)*((link.in_node)->get_active_out_td());
						node->activesum+=add_amount;
					}

				} //End for over incoming links

			} //End if ((node->type)!=SENSOR) 

		} //End for over all nodes

		// Now activate all the non-sensor nodes off their incoming activation 
		for(NNode *node: all_nodes) {
			if ((node->type)!=SENSOR) {
				//Only activate if some active input came in
				if (node->active_flag) {
					//cout<<"Activating "<<node->node_id<<" with "<<node->activesum<<": ";

					//Keep a memory of activations for potential time delayed connections
					node->last_activation2=node->last_activation;
					node->last_activation=node->activation;

					//If the node is being overrided from outside,
					//stick in the override value
					if (node->overridden()) {
						//Set activation to the override value and turn off override
						node->activate_override();
					}
					else {
						//Now run the net activation through an activation function
						if (node->ftype==SIGMOID)
							node->activation=NEAT::fsigmoid(node->activesum,4.924273,2.4621365);  //Sigmoidal activation- see comments under fsigmoid
					}
					//cout<<node->activation<<endl;

					//Increment the activation_count
					//First activation cannot be from nothing!!
					node->activation_count++;
				}
			}
		}

		onetime=true;
	}

	if (adaptable) {
        // ADAPTATION:  Adapt weights based on activations 
        for(NNode *node: all_nodes) {
            if ((node->type)!=SENSOR) {
	      
                // For each incoming connection, perform adaptation based on the trait of the connection 
                for(Link &link: node->incoming) {
		
                    if ((link.trait_id==2)||
                        (link.trait_id==3)||
                        (link.trait_id==4)) {
		  
                        //In the recurrent case we must take the last activation of the input for calculating hebbian changes
                        if (link.is_recurrent) {
                            link.weight=
                                hebbian(link.weight,maxweight,
                                        link.in_node->last_activation, 
                                        link.out_node->get_active_out(),
                                        link.params[0],link.params[1],
                                        link.params[2]);
		    
		    
                        }
                        else { //non-recurrent case
                            link.weight=
                                hebbian(link.weight,maxweight,
                                        link.in_node->get_active_out(), 
                                        link.out_node->get_active_out(),
                                        link.params[0],link.params[1],
                                        link.params[2]);
                        }
                    }
		
                }
	      
            }
	    
        }
	  
	} //end if (adaptable)

	return true;  
}

// THIS WAS NOT USED IN THE FINAL VERSION, AND NOT FULLY IMPLEMENTED,   
// BUT IT SHOWS HOW SOMETHING LIKE THIS COULD BE INITIATED
// Note that checking networks for loops in general in not necessary
// and therefore I stopped writing this function
// Check Network for loops.  Return true if its ok, false if there is a loop.
//bool Network::integrity() {
//  std::vector<NNode*>::iterator curnode;
//  std::vector<std::vector<NNode*>*> paths;
//  int count;
//  std::vector<NNode*> *newpath;
//  std::vector<std::vector<NNode*>*>::iterator curpath;

//  for(curnode=outputs.begin();curnode!=outputs.end();++curnode) {
//    newpath=new std::vector<NNode*>();
//    paths.push_back(newpath);
//    if (!((*curnode)->integrity(newpath))) return false;
//  }

//Delete the paths now that we are done
//  curpath=paths.begin();
//  for(count=0;count<paths.size();count++) {
//    delete (*curpath);
//    curpath++;
//  }

//  return true;
//}

// Prints the values of its outputs
void Network::show_activation() {
	std::vector<NNode*>::iterator curnode;
	int count;

	//if (name!=0)
	//  cout<<"Network "<<name<<" with id "<<net_id<<" outputs: (";
	//else cout<<"Network id "<<net_id<<" outputs: (";

	count=1;
	for(curnode=outputs.begin();curnode!=outputs.end();++curnode) {
		//cout<<"[Output #"<<count<<": "<<(*curnode)<<"] ";
		count++;
	}

	//cout<<")"<<endl;
}

// Takes an array of sensor values and loads it into SENSOR inputs ONLY
void Network::load_sensors(const double *sensvals) {
    for(size_t i = 0; i < inputs.size(); i++) {
        inputs[i]->sensor_load(sensvals[i]);
    }
}

void Network::load_sensors(const std::vector<double> &sensvals) {
    assert(sensvals.size() == inputs.size());

    load_sensors(sensvals.data());
}

double Network::get_output(size_t index) {
    return outputs[index]->activation;
}


// This checks a POTENTIAL link between a potential in_node and potential out_node to see if it must be recurrent 
bool Network::is_recur(NNode *potin_node,NNode *potout_node,int &count,int thresh) {
	++count;  //Count the node as visited

	if (count>thresh) {
		return false;  //Short out the whole thing- loop detected
	}

	if (potin_node==potout_node) return true;
	else {
		//Check back on all links...
        for(Link &link: potin_node->incoming) {
			//But skip links that are already recurrent
			//(We want to check back through the forward flow of signals only
			if (!link.is_recurrent) {
				if (is_recur(link.in_node,potout_node,count,thresh)) return true;
			}
		}
		return false;
	}
}

//Find the maximum number of neurons between an ouput and an input
int Network::max_depth() {
  std::vector<NNode*>::iterator curoutput; //The current output we are looking at
  int cur_depth; //The depth of the current node
  int max=0; //The max depth
  
  for(curoutput=outputs.begin();curoutput!=outputs.end();curoutput++) {
    cur_depth=(*curoutput)->depth(0,this);
    if (cur_depth>max) max=cur_depth;
  }

  return max;

}

