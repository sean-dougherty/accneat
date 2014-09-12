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
#ifndef _INNOVATION_H_
#define _INNOVATION_H_

#include "neat.h"
#include <functional>

namespace NEAT {

	enum innovtype {
		NEWNODE = 0,
		NEWLINK = 1
	};

    class InnovationId {
    public:
		innovtype innovation_type;
		int node_in_id;
		int node_out_id;
		double old_innov_num;
		bool recur_flag;

    InnovationId(int nin,
                 int nout,
                 double oldinnov)
        : innovation_type(NEWNODE)
            , node_in_id(nin)
            , node_out_id(nout)
            , old_innov_num(oldinnov) {
        }

    InnovationId(int nin,
                 int nout,
                 bool recur)
        : innovation_type(NEWLINK)
            , node_in_id(nin)
            , node_out_id(nout)
            , recur_flag(recur) {
        }

        static int cmp(const InnovationId &x, const InnovationId &y) {
#define __cmp(val)                              \
            if(x.val < y.val) {return -1;}      \
            else if(x.val > y.val) {return 1;}

            __cmp(innovation_type);
            __cmp(node_in_id);
            __cmp(node_out_id);

            switch(x.innovation_type) {
            case NEWNODE:
                __cmp(old_innov_num);
                return 0;
            case NEWLINK:
                __cmp(recur_flag);
                return 0;
            default:
                trap("invalid innovation_type");
            }

#undef __cmp
        }

        bool operator<(const InnovationId &other) const {
            return cmp(*this, other) < 0;
        }

        bool operator==(const InnovationId &other) const {
            return cmp(*this, other) == 0;
        }
    };

	// ------------------------------------------------------------
	// This Innovation class serves as a way to record innovations
	//   specifically, so that an innovation in one genome can be 
	//   compared with other innovations in the same epoch, and if they
	//   are the same innovation, they can both be assigned the same
	//   innovation number.
    //
	//  This class can encode innovations that represent a new link
	//  forming, or a new node being added.  In each case, two 
	//  nodes fully specify the innovation and where it must have
	//  occured.  (Between them)                                     
	// ------------------------------------------------------------ 
	class Innovation {
	public:
        InnovationId id;

		double innovation_num1;  //The number assigned to the innovation
		double innovation_num2;  // If this is a new node innovation, then there are 2 innovations (links) added for the new node 

		int newnode_id;  // If a new node was created, this is its node_id 

		double new_weight;   //  If a link is added, this is its weight 
		int new_trait_id; // If a link is added, this is its connected trait 

		Innovation(int nin,int nout,double num1,double num2,int newid,double oldinnov)
            : id(nin, nout, oldinnov)
            , innovation_num1(num1)
            , innovation_num2(num2)
            , newnode_id(newid) {
        }

		Innovation(int nin,int nout,double num1,double w,int t,bool recur)
            : id(nin, nout, recur)
            , innovation_num1(num1)
            , new_weight(w)
            , new_trait_id(t) {
        }
	};

    class InnovationParms {
    public:
		double new_weight;
		int new_trait_id;

        InnovationParms() {}
        InnovationParms(double w,
                        int t)
            : new_weight(w)
            , new_trait_id(t) {
        }
    };

    class IndividualInnovation {
    public:
        typedef std::function<void (Innovation *innov)> ApplyFunc;

        int population_index;
        InnovationId id;
        InnovationParms parms;
        ApplyFunc apply;

		//Constructor for the new node case
    IndividualInnovation(int population_index_,
                         InnovationId id_,
                         InnovationParms parms_,
                         ApplyFunc apply_)
        : population_index(population_index_)
            , id(id_)
            , parms(parms_)
            , apply(apply_) {
        }
    };

} // namespace NEAT

#endif
