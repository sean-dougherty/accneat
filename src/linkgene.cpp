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
#include "linkgene.h"

#include <iostream>
#include <sstream>
using namespace NEAT;

LinkGene::LinkGene(double w,
                   int inode_id,
                   int onode_id,
                   bool recur,
                   int innov,
                   double mnum) {
    _weight = w;
    _in_node_id = inode_id;
    _out_node_id = onode_id;
    _is_recurrent = recur;
    _trait_id = 1;

	innovation_num = innov;
	mutation_num = mnum;
	enable = true;
	frozen = false;
}


//Construct a gene with a trait
LinkGene::LinkGene(int trait_id,
                   double w,
                   int inode_id,
                   int onode_id,
                   bool recur,
                   int innov,
                   double mnum) {
    _weight = w;
    _in_node_id = inode_id;
    _out_node_id = onode_id;
    _is_recurrent = recur;
    _trait_id = trait_id;

	innovation_num=innov;
	mutation_num=mnum;
	enable=true;
	frozen=false;
}

LinkGene::LinkGene(LinkGene *g,
                   int trait_id,
                   int inode_id,
                   int onode_id) {
    _weight = g->_weight;
    _in_node_id = inode_id;
    _out_node_id = onode_id;
    _is_recurrent = g->_is_recurrent;
    _trait_id = trait_id;

	innovation_num=g->innovation_num;
	mutation_num=g->mutation_num;
	enable=g->enable;

	frozen=g->frozen;
}

//todo: use NodeLookup
LinkGene::LinkGene(const char *argline) {
	//LinkGene parameter holders
	int trait_id;
	int inodenum;
	int onodenum;
	double weight;
	int recur;

	//Get the gene parameters
    std::stringstream ss(argline);
    ss >> trait_id >> inodenum >> onodenum >> weight >> recur >> innovation_num >> mutation_num >> enable;

	frozen=false; //TODO: MAYBE CHANGE

    _weight = weight;
    _in_node_id = inodenum;
    _out_node_id = onodenum;
    _is_recurrent = recur;
    _trait_id = trait_id;
}

LinkGene::LinkGene(const LinkGene& gene)
{
	innovation_num = gene.innovation_num;
	mutation_num = gene.mutation_num;
	enable = gene.enable;
	frozen = gene.frozen;

    _weight = gene._weight;
    _in_node_id = gene._in_node_id;
    _out_node_id = gene._out_node_id;
    _is_recurrent = gene._is_recurrent;
    _trait_id = gene._trait_id;
}

LinkGene::~LinkGene() {
}


void LinkGene::print_to_file(std::ostream &outFile) {
	outFile<<"gene ";

	//Start off with the trait number for this gene
    outFile << _trait_id << " ";
	outFile << _in_node_id << " ";
	outFile << _out_node_id << " ";
	outFile << _weight << " ";
	outFile << _is_recurrent << " ";
	outFile << innovation_num << " ";
	outFile << mutation_num << " ";
    outFile << enable << std::endl;
}
