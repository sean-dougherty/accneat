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

#include "innovation.h"

#include <algorithm>

using namespace NEAT;
using namespace std;

static int cmp(const InnovationId &x, const InnovationId &y) {
#define __cmp(val)                              \
    if(x.val < y.val) {return -1;}              \
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

InnovationId::InnovationId(int nin,
                           int nout,
                           int oldinnov)
    : innovation_type(NEWNODE)
    , node_in_id(nin)
    , node_out_id(nout)
    , old_innov_num(oldinnov)
    , recur_flag(false) { // unused
}

InnovationId::InnovationId(int nin,
                           int nout,
                           bool recur)
    : innovation_type(NEWLINK)
    , node_in_id(nin)
    , node_out_id(nout)
    , old_innov_num(-1) // unused
    , recur_flag(recur) {
}

bool InnovationId::operator<(const InnovationId &other) const {
    return ::cmp(*this, other) < 0;
}

bool InnovationId::operator==(const InnovationId &other) const {
    return ::cmp(*this, other) == 0;
}

InnovationParms::InnovationParms()
    : new_weight(-1)
    , new_trait_id(-1) {
}

InnovationParms::InnovationParms(double w,
                                 int t)
    : new_weight(w)
    , new_trait_id(t) {
}

IndividualInnovation::IndividualInnovation(int population_index_,
                                           InnovationId id_,
                                           InnovationParms parms_,
                                           ApplyFunc apply_) 
    : population_index(population_index_)
    , id(id_)
    , parms(parms_) {
    apply = apply_;
}

// Link
Innovation::Innovation(InnovationId id_,
                       InnovationParms parms_,
                       int innovation_num1_)
    : id(id_)
    , parms(parms_)
    , innovation_num1(innovation_num1_) {
}

// Node
Innovation::Innovation(InnovationId id_,
                       InnovationParms parms_,
                       int innovation_num1_,
                       int innovation_num2_,
                       int newnode_id_)
    : id(id_)
    , parms(parms_)
    , innovation_num1(innovation_num1_)
    , innovation_num2(innovation_num2_)
    , newnode_id(newnode_id_) {
}

static bool cmp_ind(const IndividualInnovation &x, const IndividualInnovation &y) {
    return x.population_index < y.population_index;
};

void PopulationInnovations::init(int node_id, int innov_num) {
    cur_node_id = node_id;
    cur_innov_num = innov_num;
}

void PopulationInnovations::add(const IndividualInnovation &innov) {
    innovations.push_back(innov);
}

void PopulationInnovations::apply() {
    id2inds.clear();
    for(auto &ind: innovations) {
        id2inds[ind.id].push_back(ind);
    }

    vector<IndividualInnovation> masters;
    for(auto &kv: id2inds) {
        auto &inds = kv.second;
  
        sort(inds.begin(), inds.end(), cmp_ind);
  
        auto &master = inds.front();
        masters.push_back(master);
    }        

    sort(masters.begin(), masters.end(), cmp_ind);

    for(auto &master: masters) {
        auto &inds = id2inds[master.id];

        Innovation *innov;
        switch(master.id.innovation_type) {
        case NEWNODE: {
            innov = new Innovation(master.id,
                                   master.parms,
                                   cur_innov_num,
                                   cur_innov_num + 1,
                                   cur_node_id);
            cur_innov_num += 2;
            cur_node_id += 1;
        } break;
        case NEWLINK: {
            innov = new Innovation(master.id,
                                   master.parms,
                                   cur_innov_num);
            cur_innov_num += 1;
        } break;
        default:
            trap("here");
        }

        for(IndividualInnovation &ind: inds) {
            ind.apply(innov);
        }
    }

    innovations.clear();
}
