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

#include "organism.h"
#include "species.h"

using namespace NEAT;
using std::vector;

Organism::Organism(const Organism &other) {
    this->genome = other.genome->make_default();
    other.copy_into(*this);
}

Organism::Organism(const Genome &genome) {
    this->genome = genome.make_clone();

    //Note: We're in the base class constructor, so a derived class' init() won't
    //      be called. The derived class' constructor must also call init().
    init(0);
}

Organism::~Organism() {
}

void Organism::init(int gen) {
	fitness=0.0;
	generation=gen;
	error=0;
}

Organism &Organism::operator=(const Organism &other) {
    other.copy_into(*this);
    return *this;
}

void Organism::copy_into(Organism &dst) const {
#define copy(field) dst.field = this->field;
    
    copy(population_index);
    copy(fitness);
    copy(error);
    copy(net);
    *dst.genome = *this->genome;
    copy(generation);

#undef copy
}
