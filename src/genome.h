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
#pragma once

#include "rng.h"
#include <memory>

namespace NEAT {

    class Genome {
    public:
        rng_t rng;
		int genome_id;
        
        virtual ~Genome() {}


        virtual std::unique_ptr<Genome> make_default() const = 0;
        virtual Genome &operator=(const Genome &other) = 0;

        virtual void init_phenotype(class Network &net) = 0;

        virtual void print(std::ostream &out) = 0;
		virtual void verify() = 0;

        struct Stats {
            size_t nnodes;
            size_t nlinks;
        };

        virtual Stats get_stats() = 0;
    };

} // namespace NEAT


