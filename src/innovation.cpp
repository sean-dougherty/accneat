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
#include <map>

using namespace std;

namespace NEAT {

    static bool cmp_ind(const IndividualInnovation &x, const IndividualInnovation &y) {
        return x.population_index < y.population_index;
    };

    void apply_innovations(std::vector<IndividualInnovation> &allinds,
                           int *cur_node_id,
                           double *cur_innov_num) {

        map<InnovationId, vector<IndividualInnovation>> id2inds;

        for(auto &ind: allinds) {
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
                                       *cur_innov_num,
                                       *cur_innov_num + 1,
                                       *cur_node_id);
                *cur_innov_num += 2;
                *cur_node_id += 1;
            } break;
            case NEWLINK: {
                innov = new Innovation(master.id,
                                       master.parms,
                                       *cur_innov_num);
                *cur_innov_num += 1.0;
            } break;
            default:
                trap("here");
            }

            for(IndividualInnovation &ind: inds) {
                ind.apply(innov);
            }
        }

    }

}
