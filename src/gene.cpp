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
#include "gene.h"

#include <iostream>
#include <sstream>
using namespace NEAT;

Gene::Gene(double w, NNode *inode, NNode *onode, bool recur, double innov, double mnum) {
	lnk = new Link(w, inode, onode, recur);
	innovation_num = innov;
	mutation_num = mnum;

	enable = true;

	frozen = false;
}


Gene::Gene(int trait_id,double w,NNode *inode,NNode *onode,bool recur,double innov,double mnum) {
	lnk=new Link(trait_id,w,inode,onode,recur);
	innovation_num=innov;
	mutation_num=mnum;

	enable=true;

	frozen=false;
}

Gene::Gene(Gene *g,int trait_id,NNode *inode,NNode *onode) {
	lnk=new Link(trait_id,(g->lnk)->weight,inode,onode,(g->lnk)->is_recurrent);
	innovation_num=g->innovation_num;
	mutation_num=g->mutation_num;
	enable=g->enable;

	frozen=g->frozen;
}

Gene::Gene(const char *argline, std::vector<Trait*> &traits, std::vector<NNode*> &nodes) {
	//Gene parameter holders
	int trait_id;
	int inodenum;
	int onodenum;
	NNode *inode;
	NNode *onode;
	double weight;
	int recur;
	Trait *traitptr;

	std::vector<Trait*>::iterator curtrait;
	std::vector<NNode*>::iterator curnode;

	//Get the gene parameters

    std::stringstream ss(argline);
    ss >> trait_id >> inodenum >> onodenum >> weight >> recur >> innovation_num >> mutation_num >> enable;

	frozen=false; //TODO: MAYBE CHANGE

	//Get a pointer to the input node
	curnode=nodes.begin();
	while(((*curnode)->node_id)!=inodenum)
		++curnode;
	inode=(*curnode);

	//Get a pointer to the output node
	curnode=nodes.begin();
	while(((*curnode)->node_id)!=onodenum)
		++curnode;
	onode=(*curnode);

	lnk=new Link(trait_id,weight,inode,onode,recur);
}

Gene::Gene(const Gene& gene)
{
	innovation_num = gene.innovation_num;
	mutation_num = gene.mutation_num;
	enable = gene.enable;
	frozen = gene.frozen;

	lnk = new Link(*gene.lnk);
}

Gene::~Gene() {
	delete lnk;
}


void Gene::print_to_file(std::ostream &outFile) {
	outFile<<"gene ";

	//Start off with the trait number for this gene
    outFile<<lnk->get_trait_id()<<" ";
	outFile<<(lnk->in_node)->node_id<<" ";
	outFile<<(lnk->out_node)->node_id<<" ";
	outFile<<(lnk->weight)<<" ";
	outFile<<(lnk->is_recurrent)<<" ";
	outFile<<innovation_num<<" ";
	outFile<<mutation_num<<" ";
    outFile<<enable<<std::endl;
}
