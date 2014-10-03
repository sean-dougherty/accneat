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
#include "innovgenome.h"

#include "protoinnovlinkgene.h"
#include "recurrencychecker.h"
#include "util.h"
#include <assert.h>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <sstream>

using namespace NEAT;
using namespace std;

void InnovGenome::reset() {
    traits.clear();
    nodes.clear();
    links.clear();
}

InnovGenome::InnovGenome()
    : node_lookup(nodes) {
}

InnovGenome::InnovGenome(rng_t rng_,
                         size_t ntraits,
                         size_t ninputs,
                         size_t noutputs,
                         size_t nhidden)
    : InnovGenome() {

    rng = rng_;

    for(size_t i = 0; i < ntraits; i++) {
        traits.emplace_back(i + 1,
                            rng.prob(),
                            rng.prob(),
                            rng.prob(),
                            rng.prob(),
                            rng.prob(),
                            rng.prob(),
                            rng.prob(),
                            rng.prob(),
                            rng.prob());
    }

    {
        int node_id = 1;

        //Bias node
        add_node(nodes, InnovNodeGene(nodetype::BIAS, node_id++));

        //Sensor nodes
        for(size_t i = 0; i < ninputs; i++) {
            add_node(nodes, InnovNodeGene(nodetype::SENSOR, node_id++));
        }

        //Output nodes
        for(size_t i = 0; i < noutputs; i++) {
            add_node(nodes, InnovNodeGene(nodetype::OUTPUT, node_id++));
        }

        //Hidden nodes
        for(size_t i = 0; i < nhidden; i++) {
            add_node(nodes, InnovNodeGene(nodetype::HIDDEN, node_id++));
        }
    }

    const int node_id_bias = 1;
    const int node_id_input = node_id_bias + 1;
    const int node_id_output = node_id_input + ninputs;
    const int node_id_hidden = node_id_output + noutputs;

    assert(nhidden > 0);

    int innov = 1;

    //Create links from Bias to all hidden
    for(size_t i = 0; i < nhidden; i++) {
        add_link( links, InnovLinkGene(rng.element(traits).trait_id,
                                       rng.prob(),
                                       node_id_bias,
                                       i + node_id_hidden,
                                       false,
                                       innov++,
                                       0.0) );
    }

    //Create links from all inputs to all hidden
    for(size_t i = 0; i < ninputs; i++) {
        for(size_t j = 0; j < nhidden; j++) {
            add_link( links, InnovLinkGene(rng.element(traits).trait_id,
                                           rng.prob(),
                                           i + node_id_input,
                                           j + node_id_hidden,
                                           false,
                                           innov++,
                                           0.0));
        }
    }

    //Create links from all hidden to all output
    for(size_t i = 0; i < nhidden; i++) {
        for(size_t j = 0; j < noutputs; j++) {
            add_link( links, InnovLinkGene(rng.element(traits).trait_id,
                                           rng.prob(),
                                           i + node_id_hidden,
                                           j + node_id_output,
                                           false,
                                           innov++,
                                           0.0));
        }
    }
}

unique_ptr<Genome> InnovGenome::make_default() const {
    return unique_ptr<Genome>(new InnovGenome());
}

Genome &InnovGenome::operator=(const Genome &other) {
    return *this = dynamic_cast<const InnovGenome &>(other);
}

InnovGenome::~InnovGenome() {
}

void InnovGenome::verify() {
#ifdef NDEBUG
    return;
#else

	//Check for InnovNodeGenes being out of order
    for(size_t i = 1, n = nodes.size(); i < n; i++) {
        assert( nodes[i-1].node_id < nodes[i].node_id );
    }

    {
        //Check links reference valid nodes.
        for(InnovLinkGene &gene: links) {
            assert( get_node(gene.in_node_id()) );
            assert( get_node(gene.out_node_id()) );
        }
    }

	//Make sure there are no duplicate genes
	for(InnovLinkGene &gene: links) {
		for(InnovLinkGene &gene2: links) {
            if(&gene != &gene2) {
                assert( (gene.is_recurrent() != gene2.is_recurrent())
                        || (gene2.in_node_id() != gene.in_node_id())
                        || (gene2.out_node_id() != gene.out_node_id()) );
            }
		}
	}
#endif
}

Genome::Stats InnovGenome::get_stats() {
    return {nodes.size(), links.size()};
}

void InnovGenome::print(std::ostream &out) {
    out<<"genomestart "<<genome_id<<std::endl;

	//Output the traits
    for(auto &t: traits)
        t.print_to_file(out);

    //Output the nodes
    for(auto &n: nodes)
        n.print_to_file(out);

    //Output the genes
    for(auto &g: links)
        g.print_to_file(out);

    out << "genomeend " << genome_id << std::endl;
}

int InnovGenome::get_last_node_id() {
    return nodes.back().node_id + 1;
}

real_t InnovGenome::get_last_gene_innovnum() {
    return links.back().innovation_num + 1;
}

void InnovGenome::duplicate_into(InnovGenome *offspring) const {
    offspring->traits = traits;
    offspring->links = links;
    offspring->nodes = nodes;
}

InnovGenome &InnovGenome::operator=(const InnovGenome &other) {
    rng = other.rng;
    genome_id = other.genome_id;
    traits = other.traits;
    nodes = other.nodes;
    links = other.links;
    return *this;
}

void InnovGenome::mutate_random_trait() {
    rng.element(traits).mutate(rng);
}

void InnovGenome::mutate_link_trait(int times) {
    for(int i = 0; i < times; i++) {
        int trait_id = 1 + rng.index(traits);
        InnovLinkGene &gene = rng.element(links);
        
        if(!gene.frozen) {
            gene.set_trait_id(trait_id);
        }
    }
}

void InnovGenome::mutate_node_trait(int times) {
    for(int i = 0; i < times; i++) {
        int trait_id = 1 + rng.index(traits);
        InnovNodeGene &node = rng.element(nodes);

        if(!node.frozen) {
            node.set_trait_id(trait_id);
        }
    }

    //TRACK INNOVATION! - possible future use
    //for any gene involving the mutated node, perturb that gene's
    //mutation number
}

void InnovGenome::mutate_link_weights(real_t power,real_t rate,mutator mut_type) {
	//Go through all the InnovLinkGenes and perturb their link's weights

	real_t num = 0.0; //counts gene placement
	real_t gene_total = (real_t)links.size();
	real_t endpart = gene_total*0.8; //Signifies the last part of the genome
	real_t powermod = 1.0; //Modified power by gene number
	//The power of mutation will rise farther into the genome
	//on the theory that the older genes are more fit since
	//they have stood the test of time

	bool severe = rng.prob() > 0.5;  //Once in a while really shake things up

	//Loop on all links  (ORIGINAL METHOD)
	for(InnovLinkGene &gene: links) {

		//The following if determines the probabilities of doing cold gaussian
		//mutation, meaning the probability of replacing a link weight with
		//another, entirely random weight.  It is meant to bias such mutations
		//to the tail of a genome, because that is where less time-tested links
		//reside.  The gausspoint and coldgausspoint represent values above
		//which a random float will signify that kind of mutation.  

		//Don't mutate weights of frozen links
		if (!(gene.frozen)) {
            real_t gausspoint;
            real_t coldgausspoint;

			if (severe) {
				gausspoint=0.3;
				coldgausspoint=0.1;
			}
			else if ((gene_total>=10.0)&&(num>endpart)) {
				gausspoint=0.5;  //Mutate by modification % of connections
				coldgausspoint=0.3; //Mutate the rest by replacement % of the time
			}
			else {
				//Half the time don't do any cold mutations
				if (rng.prob()>0.5) {
					gausspoint=1.0-rate;
					coldgausspoint=1.0-rate-0.1;
				}
				else {
					gausspoint=1.0-rate;
					coldgausspoint=1.0-rate;
				}
			}

			//Possible methods of setting the perturbation:
			real_t randnum = rng.posneg()*rng.prob()*power*powermod;
			if (mut_type==GAUSSIAN) {
				real_t randchoice = rng.prob();
				if (randchoice > gausspoint)
					gene.weight()+=randnum;
				else if (randchoice > coldgausspoint)
					gene.weight()=randnum;
			}
			else if (mut_type==COLDGAUSSIAN)
				gene.weight()=randnum;

			//Cap the weights at 8.0 (experimental)
			if (gene.weight() > 8.0) gene.weight() = 8.0;
			else if (gene.weight() < -8.0) gene.weight() = -8.0;

			//Record the innovation
			gene.mutation_num = gene.weight();

			num+=1.0;
		}

	} //end for loop
}

void InnovGenome::mutate_toggle_enable(int times) {
    assert(NEAT::search_type == GeneticSearchType::COMPLEXIFY);

    for(int i = 0; i < times; i++) {
        InnovLinkGene &gene = rng.element(links);

        if(!gene.enable) {
            gene.enable = true;
        } else {
			//We need to make sure that another gene connects out of the in-node
			//Because if not a section of network will break off and become isolated
            bool found = false;
            for(InnovLinkGene &checkgene: links) {
                if( (checkgene.in_node_id() == gene.in_node_id())
                    && checkgene.enable
                    && (checkgene.innovation_num != gene.innovation_num) ) {
                    found = true;
                    break;
                }
            }

			//Disable the gene if it's safe to do so
			if(found)
				gene.enable = false;
        }
    }
}

void InnovGenome::mutate_gene_reenable() {
    assert(NEAT::search_type == GeneticSearchType::COMPLEXIFY);

	//Search for a disabled gene
    for(InnovLinkGene &g: links) {
        if(!g.enable) {
            g.enable = true;
            break;
        }
    }
}

bool InnovGenome::mutate_add_node(CreateInnovationFunc create_innov) {
    InnovLinkGene *splitlink = nullptr;
    {
        for(int i = 0; !splitlink && i < 20; i++) {
            InnovLinkGene &g = rng.element(links);
            //If either the link is disabled, or it has a bias input, try again
            if( g.enable && get_node(g.in_node_id())->type != nodetype::BIAS ) {
                splitlink = &g;
            }
        }
        //We couldn't find anything, so say goodbye!
        if (!splitlink) {
            return false;
        }
    }

    InnovationId innov_id(splitlink->in_node_id(),
                          splitlink->out_node_id(),
                          splitlink->innovation_num);
    InnovationParms innov_parms;

    auto innov_apply = [this, splitlink] (const Innovation *innov) {

        InnovNodeGene newnode(nodetype::HIDDEN, innov->newnode_id);

        InnovLinkGene newlink1(splitlink->trait_id(),
                               1.0,
                               innov->id.node_in_id,
                               innov->newnode_id,
                               splitlink->is_recurrent(),
                               innov->innovation_num1,
                               0);

        InnovLinkGene newlink2(splitlink->trait_id(),
                               splitlink->weight(),
                               innov->newnode_id,
                               innov->id.node_out_id,
                               false,
                               innov->innovation_num2,
                               0);    

        // If deletion of links is permitted, delete it.
        if(NEAT::search_type == GeneticSearchType::COMPLEXIFY) {
            splitlink->enable = false;
        } else {
            delete_link(splitlink);
        }

        add_link(this->links, newlink1);
        add_link(this->links, newlink2);
        add_node(this->nodes, newnode);
    };

    create_innov(innov_id, innov_parms, innov_apply);

	return true;
}

void InnovGenome::mutate_delete_node() {
    size_t first_non_io;
    for(first_non_io = 0; first_non_io < nodes.size(); first_non_io++) {
        if( nodes[first_non_io].type == nodetype::HIDDEN ) {
            break;
        }
    }

    //Don't delete if only 0 or 1 hidden nodes
    if(first_non_io >= (nodes.size()-1)) {
        return;
    }

    size_t node_index = rng.index(nodes, first_non_io);
    InnovNodeGene node = nodes[node_index];
    assert(node.type == nodetype::HIDDEN);

    nodes.erase(nodes.begin() + node_index);

    //todo: we should have a way to look up links by in/out id
    auto it_end = std::remove_if(links.begin(), links.end(),
                                 [&node] (const InnovLinkGene &link) {
                                     return link.in_node_id() == node.node_id
                                     || link.out_node_id() == node.node_id;
                                 });

    links.resize(it_end - links.begin());
}

void InnovGenome::mutate_delete_link() {
    if(links.size() <= 1)
        return;

    size_t link_index = rng.index(links);
    InnovLinkGene link = links[link_index];
    links.erase(links.begin() + link_index);

    delete_if_orphaned_hidden_node(link.in_node_id());
    delete_if_orphaned_hidden_node(link.out_node_id());
}

bool InnovGenome::mutate_add_link(CreateInnovationFunc create_innov,
                                  int tries) {
    InnovLinkGene *recur_checker_buf[links.size()];
    RecurrencyChecker recur_checker(nodes.size(), links, recur_checker_buf);

	InnovNodeGene *in_node = nullptr; //Pointers to the nodes
	InnovNodeGene *out_node = nullptr; //Pointers to the nodes

	//Decide whether to make this recurrent
	bool do_recur = rng.prob() < NEAT::recur_only_prob;

    // Try to find nodes for link.
    {
        bool found_nodes = false;

        //Find the first non-sensor so that the to-node won't look at sensors as
        //possible destinations
        int first_nonsensor = 0;
        for(; is_input(nodes[first_nonsensor].get_type()); first_nonsensor++) {
        }

        for(int trycount = 0; !found_nodes && (trycount < tries); trycount++) {
            //Here is the recurrent finder loop- it is done separately
            if(do_recur) {
                //Some of the time try to make a recur loop
                // todo: make this an NE parm?
                if (rng.prob() > 0.5) {
                    in_node = &rng.element(nodes, first_nonsensor);
                    out_node = in_node;
                }
                else {
                    //Choose random nodenums
                    in_node = &rng.element(nodes);
                    out_node = &rng.element(nodes, first_nonsensor);
                }
            } else {
                //Choose random nodenums
                in_node = &rng.element(nodes);
                out_node = &rng.element(nodes, first_nonsensor);
            }

            InnovLinkGene *existing_link = find_link(in_node->node_id, out_node->node_id, do_recur);
            if(existing_link != nullptr) {
                if( NEAT::mutate_add_link_reenables ) {
                    existing_link->enable = true;
                    return true;
                }
            } else if(do_recur == recur_checker.is_recur(in_node->node_id,
                                                         out_node->node_id)) {
                found_nodes = true;
            }
        }

        assert( !is_input(out_node->type) );

        //Continue only if an open link was found
        if(!found_nodes) {
            return false;
        }
    }

    // Create the gene.
    {
        InnovationId innov_id(in_node->node_id,
                              out_node->node_id,
                              do_recur);

        //These two values may or may not take effect in the new innovation.
        //It depends on whether this genome is the first to create the innovation,
        //but it's impossible to know at this point who is first.
        int trait_id = 1 + rng.index(traits);
        real_t newweight = rng.posneg() * rng.prob() * 1.0;

        InnovationParms innov_parms(newweight, trait_id);

        auto innov_apply = [this] (const Innovation *innov) {

            InnovLinkGene newlink(innov->parms.new_trait_id,
                                  innov->parms.new_weight,
                                  innov->id.node_in_id,
                                  innov->id.node_out_id,
                                  innov->id.recur_flag,
                                  innov->innovation_num1,
                                  innov->parms.new_weight);

            add_link(this->links, newlink);
        };

        create_innov(innov_id, innov_parms, innov_apply);
    }

    return true;
}

void InnovGenome::add_link(vector<InnovLinkGene> &llist, const InnovLinkGene &l) {
    auto it = std::upper_bound(llist.begin(), llist.end(), l, linklist_cmp);
    llist.insert(it, l);
}

void InnovGenome::add_node(vector<InnovNodeGene> &nlist, const InnovNodeGene &n) {
    auto it = std::upper_bound(nlist.begin(), nlist.end(), n, nodelist_cmp);
    nlist.insert(it, n);
}

void InnovGenome::mate(InnovGenome *genome1,
                       InnovGenome *genome2,
                       InnovGenome *offspring,
                       real_t fitness1,
                       real_t fitness2) {

    //Perform mating based on probabilities of differrent mating types
    if( offspring->rng.prob() < NEAT::mate_multipoint_prob ) { 
        InnovGenome::mate_multipoint(genome1,
                                     genome2,
                                     offspring,
                                     fitness1,
                                     fitness2);
    } else {
        InnovGenome::mate_multipoint_avg(genome1,
                                         genome2,
                                         offspring,
                                         fitness1,
                                         fitness2);
    }
}

// todo: use NodeLookup for newnodes instead of linear search!
void InnovGenome::mate_multipoint(InnovGenome *genome1,
                                  InnovGenome *genome2,
                                  InnovGenome *offspring,
                                  real_t fitness1,
                                  real_t fitness2) {
    rng_t &rng = offspring->rng;
    vector<InnovLinkGene> &links1 = genome1->links;
    vector<InnovLinkGene> &links2 = genome2->links;

	//The baby InnovGenome will contain these new Traits, InnovNodeGenes, and InnovLinkGenes
    offspring->reset();
	vector<Trait> &newtraits = offspring->traits;
	vector<InnovNodeGene> &newnodes = offspring->nodes;   
	vector<InnovLinkGene> &newlinks = offspring->links;    

	vector<InnovLinkGene>::iterator curgene2;  //Checks for link duplication

	//iterators for moving through the two parents' traits
	vector<Trait*>::iterator p1trait;
	vector<Trait*>::iterator p2trait;

	//iterators for moving through the two parents' links
	vector<InnovLinkGene>::iterator p1gene;
	vector<InnovLinkGene>::iterator p2gene;
	real_t p1innov;  //Innovation numbers for links inside parents' InnovGenomes
	real_t p2innov;
	vector<InnovNodeGene>::iterator curnode;  //For checking if InnovNodeGenes exist already 

	bool disable;  //Set to true if we want to disabled a chosen gene

	disable=false;
	InnovLinkGene newgene;

	bool p1better; //Tells if the first genome (this one) has better fitness or not

	bool skip;

	//First, average the Traits from the 2 parents to form the baby's Traits
	//It is assumed that trait lists are the same length
	//In the future, may decide on a different method for trait mating
    assert(genome1->traits.size() == genome2->traits.size());
    for(size_t i = 0, n = genome1->traits.size(); i < n; i++) {
        newtraits.emplace_back(genome1->traits[i], genome2->traits[i]);
    }

	//Figure out which genome is better
	//The worse genome should not be allowed to add extra structural baggage
	//If they are the same, use the smaller one's disjoint and excess genes only
	if (fitness1>fitness2) 
		p1better=true;
	else if (fitness1==fitness2) {
		if (links1.size()<(links2.size()))
			p1better=true;
		else p1better=false;
	}
	else 
		p1better=false;

	//Make sure all sensors and outputs are included
    for(InnovNodeGene &node: genome1->nodes) {
		if(node.type != nodetype::HIDDEN) {
            //Add the new node
            add_node(newnodes, node);
        } else {
            break;
        }
    }

	//Now move through the InnovLinkGenes of each parent until both genomes end
	p1gene = links1.begin();
	p2gene = links2.begin();
	while( !((p1gene==links1.end()) && (p2gene==(links2).end())) ) {
        ProtoInnovLinkGene protogene;

        skip=false;  //Default to not skipping a chosen gene

        if (p1gene==links1.end()) {
            protogene.set_gene(genome2, &*p2gene);
            ++p2gene;
            if (p1better) skip=true;  //Skip excess from the worse genome
        } else if (p2gene==(links2).end()) {
            protogene.set_gene(genome1, &*p1gene);
            ++p1gene;
            if (!p1better) skip=true; //Skip excess from the worse genome
        } else {
            //Extract current innovation numbers
            p1innov=p1gene->innovation_num;
            p2innov=p2gene->innovation_num;

            if (p1innov==p2innov) {
                if (rng.prob()<0.5) {
                    protogene.set_gene(genome1, &*p1gene);
                } else {
                    protogene.set_gene(genome2, &*p2gene);
                }

                //If one is disabled, the corresponding gene in the offspring
                //will likely be disabled
                if (((p1gene->enable)==false)||
                    ((p2gene->enable)==false)) 
                    if (rng.prob()<0.75) disable=true;

                ++p1gene;
                ++p2gene;
            } else if (p1innov < p2innov) {
                protogene.set_gene(genome1, &*p1gene);
                ++p1gene;

                if (!p1better) skip=true;

            } else if (p2innov<p1innov) {
                protogene.set_gene(genome2, &*p2gene);
                ++p2gene;

                if (p1better) skip=true;
            }
        }

        //Check to see if the protogene conflicts with an already chosen gene
        //i.e. do they represent the same link    
        curgene2=newlinks.begin();
        while ((curgene2!=newlinks.end())&&
               (!((curgene2->in_node_id()==protogene.gene()->in_node_id())&&
                  (curgene2->out_node_id()==protogene.gene()->out_node_id())&&(curgene2->is_recurrent()== protogene.gene()->is_recurrent()) ))&&
               (!((curgene2->in_node_id()==protogene.gene()->out_node_id())&&
                  (curgene2->out_node_id()==protogene.gene()->in_node_id())&&
                  (!(curgene2->is_recurrent()))&&
                  (!(protogene.gene()->is_recurrent())) )))
        {	
            ++curgene2;
        }

        if (curgene2!=newlinks.end()) skip=true;  //Links conflicts, abort adding

        if (!skip) {
            //Now add the gene to the baby
            InnovNodeGene new_inode;
            InnovNodeGene new_onode;

            //Next check for the nodes, add them if not in the baby InnovGenome already
            InnovNodeGene *inode = protogene.in();
            InnovNodeGene *onode = protogene.out();

            //Check for inode in the newnodes list
            if (inode->node_id<onode->node_id) {
                //inode before onode

                //Checking for inode's existence
                curnode=newnodes.begin();
                while((curnode!=newnodes.end())&&
                      (curnode->node_id!=inode->node_id)) 
                    ++curnode;

                if (curnode==newnodes.end()) {
                    //Here we know the node doesn't exist so we have to add it
                    new_inode = *inode;
                    add_node(newnodes,new_inode);

                }
                else {
                    new_inode=*curnode;

                }

                //Checking for onode's existence
                curnode=newnodes.begin();
                while((curnode!=newnodes.end())&&
                      (curnode->node_id!=onode->node_id)) 
                    ++curnode;
                if (curnode==newnodes.end()) {
                    //Here we know the node doesn't exist so we have to add it
                    new_onode = *onode;
                    add_node(newnodes,new_onode);

                }
                else {
                    new_onode=*curnode;
                }

            }
            //If the onode has a higher id than the inode we want to add it first
            else {
                //Checking for onode's existence
                curnode=newnodes.begin();
                while((curnode!=newnodes.end())&&
                      (curnode->node_id!=onode->node_id)) 
                    ++curnode;
                if (curnode==newnodes.end()) {
                    //Here we know the node doesn't exist so we have to add it
                    new_onode = *onode;
                    //newnodes.push_back(new_onode);
                    add_node(newnodes,new_onode);

                }
                else {
                    new_onode=*curnode;

                }

                //Checking for inode's existence
                curnode=newnodes.begin();
                while((curnode!=newnodes.end())&&
                      (curnode->node_id!=inode->node_id)) 
                    ++curnode;
                if (curnode==newnodes.end()) {
                    //Here we know the node doesn't exist so we have to add it
                    new_inode = *inode;
                    add_node(newnodes,new_inode);
                }
                else {
                    new_inode=*curnode;

                }

            } //End InnovNodeGene checking section- InnovNodeGenes are now in new InnovGenome

            //Add the InnovLinkGene
            newgene = InnovLinkGene(protogene.gene(),
                                    protogene.gene()->trait_id(),
                                    new_inode.node_id,
                                    new_onode.node_id);
            if (disable) {
                newgene.enable=false;
                disable=false;
            }
            newlinks.push_back(newgene);
        }

    }
}

// todo: use NodeLookup for newnodes instead of linear search!
void InnovGenome::mate_multipoint_avg(InnovGenome *genome1,
                                      InnovGenome *genome2,
                                      InnovGenome *offspring,
                                      real_t fitness1,
                                      real_t fitness2) {
    rng_t &rng = offspring->rng;
    vector<InnovLinkGene> &links1 = genome1->links;
    vector<InnovLinkGene> &links2 = genome2->links;

	//The baby InnovGenome will contain these new Traits, InnovNodeGenes, and InnovLinkGenes
    offspring->reset();
	vector<Trait> &newtraits = offspring->traits;
	vector<InnovNodeGene> &newnodes = offspring->nodes;
	vector<InnovLinkGene> &newlinks = offspring->links;

	vector<InnovLinkGene>::iterator curgene2; //Checking for link duplication

	//iterators for moving through the two parents' links
	vector<InnovLinkGene>::iterator p1gene;
	vector<InnovLinkGene>::iterator p2gene;
	real_t p1innov;  //Innovation numbers for links inside parents' InnovGenomes
	real_t p2innov;
	vector<InnovNodeGene>::iterator curnode;  //For checking if InnovNodeGenes exist already 

	//This InnovLinkGene is used to hold the average of the two links to be averaged
	InnovLinkGene avgene(0,0,0,0,0,0,0);
	InnovLinkGene newgene;

	bool skip;

	bool p1better;  //Designate the better genome

	//First, average the Traits from the 2 parents to form the baby's Traits
	//It is assumed that trait lists are the same length
	//In future, could be done differently
    for(size_t i = 0, n = genome1->traits.size(); i < n; i++) {
        newtraits.emplace_back(genome1->traits[i], genome2->traits[i]);
	}

	//NEW 3/17/03 Make sure all sensors and outputs are included
    for(InnovNodeGene &node: genome1->nodes) {
		if(node.type != nodetype::HIDDEN) {
            add_node(newnodes, node);
        } else {
            break;
        }
	}

	//Figure out which genome is better
	//The worse genome should not be allowed to add extra structural baggage
	//If they are the same, use the smaller one's disjoint and excess genes only
	if (fitness1>fitness2) 
		p1better=true;
	else if (fitness1==fitness2) {
		if (links1.size()<(links2.size()))
			p1better=true;
		else p1better=false;
	}
	else 
		p1better=false;


	//Now move through the InnovLinkGenes of each parent until both genomes end
	p1gene=links1.begin();
	p2gene=links2.begin();
	while(!((p1gene==links1.end()) && (p2gene==(links2).end()))) {
        ProtoInnovLinkGene protogene;

        avgene.enable=true;  //Default to enabled

        skip=false;

        if (p1gene==links1.end()) {
            protogene.set_gene(genome2, &*p2gene);
            ++p2gene;

            if (p1better) skip=true;

        }
        else if (p2gene==(links2).end()) {
            protogene.set_gene(genome1, &*p1gene);
            ++p1gene;

            if (!p1better) skip=true;
        }
        else {
            //Extract current innovation numbers
            p1innov=p1gene->innovation_num;
            p2innov=p2gene->innovation_num;

            if (p1innov==p2innov) {
                protogene.set_gene(nullptr, &avgene);

                //Average them into the avgene
                if (rng.prob()>0.5) {
                    avgene.set_trait_id(p1gene->trait_id());
                } else {
                    avgene.set_trait_id(p2gene->trait_id());
                }

                //WEIGHTS AVERAGED HERE
                avgene.weight() = (p1gene->weight()+p2gene->weight())/2.0;

                if(rng.prob() > 0.5) {
                    protogene.set_in(genome1->get_node(p1gene->in_node_id()));
                } else {
                    protogene.set_in(genome2->get_node(p2gene->in_node_id()));
                }

                if(rng.prob() > 0.5) {
                    protogene.set_out(genome1->get_node(p1gene->out_node_id()));
                } else {
                    protogene.set_out(genome2->get_node(p2gene->out_node_id()));
                }

                if (rng.prob()>0.5) avgene.set_recurrent(p1gene->is_recurrent());
                else avgene.set_recurrent(p2gene->is_recurrent());

                avgene.innovation_num=p1gene->innovation_num;
                avgene.mutation_num=(p1gene->mutation_num+p2gene->mutation_num)/2.0;

                if (((p1gene->enable)==false)||
                    ((p2gene->enable)==false)) 
                    if (rng.prob()<0.75) avgene.enable=false;

                ++p1gene;
                ++p2gene;
            } else if (p1innov<p2innov) {
                protogene.set_gene(genome1, &*p1gene);
                ++p1gene;

                if (!p1better) skip=true;
            } else if (p2innov<p1innov) {
                protogene.set_gene(genome2, &*p2gene);
                ++p2gene;

                if (p1better) skip=true;
            }
        }

        //Check to see if the chosengene conflicts with an already chosen gene
        //i.e. do they represent the same link    
        curgene2=newlinks.begin();
        while ((curgene2!=newlinks.end()))

        {

            if (((curgene2->in_node_id()==protogene.gene()->in_node_id())&&
                 (curgene2->out_node_id()==protogene.gene()->out_node_id())&&
                 (curgene2->is_recurrent()== protogene.gene()->is_recurrent()))||
                ((curgene2->out_node_id()==protogene.gene()->in_node_id())&&
                 (curgene2->in_node_id()==protogene.gene()->out_node_id())&&
                 (!(curgene2->is_recurrent()))&&
                 (!(protogene.gene()->is_recurrent()))     ))
            { 
                skip=true;

            }
            ++curgene2;
        }

        if (!skip) {
            //Now add the chosengene to the baby

            //Next check for the nodes, add them if not in the baby InnovGenome already
            InnovNodeGene *inode = protogene.in();
            InnovNodeGene *onode = protogene.out();

            //Check for inode in the newnodes list
            InnovNodeGene new_inode;
            InnovNodeGene new_onode;
            if (inode->node_id<onode->node_id) {

                //Checking for inode's existence
                curnode=newnodes.begin();
                while((curnode!=newnodes.end())&&
                      (curnode->node_id!=inode->node_id)) 
                    ++curnode;

                if (curnode==newnodes.end()) {
                    //Here we know the node doesn't exist so we have to add it
                    new_inode = *inode;
                    add_node(newnodes,new_inode);
                }
                else {
                    new_inode=(*curnode);

                }

                //Checking for onode's existence
                curnode=newnodes.begin();
                while((curnode!=newnodes.end())&&
                      (curnode->node_id!=onode->node_id)) 
                    ++curnode;
                if (curnode==newnodes.end()) {
                    //Here we know the node doesn't exist so we have to add it
                    new_onode = *onode;

                    add_node(newnodes,new_onode);
                }
                else {
                    new_onode=(*curnode);
                }
            }
            //If the onode has a higher id than the inode we want to add it first
            else {
                //Checking for onode's existence
                curnode=newnodes.begin();
                while((curnode!=newnodes.end())&&
                      (curnode->node_id!=onode->node_id)) 
                    ++curnode;
                if (curnode==newnodes.end()) {
                    //Here we know the node doesn't exist so we have to add it
                    new_onode = *onode;

                    add_node(newnodes,new_onode);
                }
                else {
                    new_onode=(*curnode);
                }

                //Checking for inode's existence
                curnode=newnodes.begin();
                while((curnode!=newnodes.end())&&
                      (curnode->node_id!=inode->node_id)) 
                    ++curnode;
                if (curnode==newnodes.end()) {
                    //Here we know the node doesn't exist so we have to add it
                    new_inode = *inode;

                    add_node(newnodes,new_inode);
                }
                else {
                    new_inode=(*curnode);

                }

            } //End InnovNodeGene checking section- InnovNodeGenes are now in new InnovGenome

            //Add the InnovLinkGene
            newgene = InnovLinkGene(protogene.gene(),
                                    protogene.gene()->trait_id(),
                                    new_inode.node_id,
                                    new_onode.node_id);

            newlinks.push_back(newgene);

        }  //End if which checked for link duplicationb

    }
}

real_t InnovGenome::compatibility(InnovGenome *g) {
    vector<InnovLinkGene> &links1 = this->links;
    vector<InnovLinkGene> &links2 = g->links;


	//Innovation numbers
	real_t p1innov;
	real_t p2innov;

	//Intermediate value
	real_t mut_diff;

	//Set up the counters
	real_t num_disjoint=0.0;
	real_t num_excess=0.0;
	real_t mut_diff_total=0.0;
	real_t num_matching=0.0;  //Used to normalize mutation_num differences

	//Now move through the InnovLinkGenes of each potential parent 
	//until both InnovGenomes end
	vector<InnovLinkGene>::iterator p1gene = links1.begin();
	vector<InnovLinkGene>::iterator p2gene = links2.begin();

	while(!((p1gene==links1.end())&&
            (p2gene==links2.end()))) {

        if (p1gene==links1.end()) {
            ++p2gene;
            num_excess+=1.0;
        }
        else if (p2gene==links2.end()) {
            ++p1gene;
            num_excess+=1.0;
        }
        else {
            //Extract current innovation numbers
            p1innov = p1gene->innovation_num;
            p2innov = p2gene->innovation_num;

            if (p1innov==p2innov) {
                num_matching+=1.0;
                mut_diff = p1gene->mutation_num - p2gene->mutation_num;
                if (mut_diff<0.0) mut_diff=0.0-mut_diff;
                mut_diff_total+=mut_diff;

                ++p1gene;
                ++p2gene;
            }
            else if (p1innov<p2innov) {
                ++p1gene;
                num_disjoint+=1.0;
            }
            else if (p2innov<p1innov) {
                ++p2gene;
                num_disjoint+=1.0;
            }
        }
    } //End while

    //Return the compatibility number using compatibility formula
    //Note that mut_diff_total/num_matching gives the AVERAGE
    //difference between mutation_nums for any two matching InnovLinkGenes
    //in the InnovGenome

    //Normalizing for genome size
    //return (disjoint_coeff*(num_disjoint/max_genome_size)+
    //  excess_coeff*(num_excess/max_genome_size)+
    //  mutdiff_coeff*(mut_diff_total/num_matching));


    //Look at disjointedness and excess in the absolute (ignoring size)

    return (NEAT::disjoint_coeff*(num_disjoint/1.0)+
			NEAT::excess_coeff*(num_excess/1.0)+
			NEAT::mutdiff_coeff*(mut_diff_total/num_matching));
}

real_t InnovGenome::trait_compare(Trait *t1,Trait *t2) {

	int id1=t1->trait_id;
	int id2=t2->trait_id;
	int count;
	real_t params_diff=0.0; //Measures parameter difference

	//See if traits represent different fundamental types of connections
	if ((id1==1)&&(id2>=2)) {
		return 0.5;
	}
	else if ((id2==1)&&(id1>=2)) {
		return 0.5;
	}
	//Otherwise, when types are same, compare the actual parameters
	else {
		if (id1>=2) {
			for (count=0;count<=2;count++) {
				params_diff += fabs(t1->params[count]-t2->params[count]);
			}
			return params_diff/4.0;
		}
		else return 0.0; //For type 1, params are not applicable
	}

}

void InnovGenome::randomize_traits() {
    for(InnovNodeGene &node: nodes) {
		node.set_trait_id(1 + rng.index(traits));
	}

    for(InnovLinkGene &gene: links) {
		gene.set_trait_id(1 + rng.index(traits));
	}
}

inline Trait &get_trait(vector<Trait> &traits, int trait_id) {
    Trait &t = traits[trait_id - 1];
    assert(t.trait_id == trait_id);
    return t;
}

Trait &InnovGenome::get_trait(const InnovNodeGene &node) {
    return ::get_trait(traits, node.get_trait_id());
}

Trait &InnovGenome::get_trait(const InnovLinkGene &gene) {
    return ::get_trait(traits, gene.trait_id());
}

void InnovGenome::init_phenotype(Network &net) {
	real_t maxweight=0.0; //Compute the maximum weight for adaptation purposes
	real_t weight_mag; //Measures absolute value of weights

    net.reset();
    vector<NNode> &netnodes = net.nodes;

	//Create the nodes
	for(InnovNodeGene &node: nodes) {
        netnodes.emplace_back(node.type);
	}

	//Create the links by iterating through the genes
    for(InnovLinkGene &gene: links) {
		//Only create the link if the gene is enabled
		if(gene.enable) {
            node_index_t inode = get_node_index(gene.in_node_id());
            node_index_t onode = get_node_index(gene.out_node_id());

			//NOTE: This line could be run through a recurrency check if desired
			// (no need to in the current implementation of NEAT)
			netnodes[onode].incoming.emplace_back(gene.weight(), inode);

            Link &newlink = netnodes[onode].incoming.back();

			//Keep track of maximum weight
			if (newlink.weight>0)
				weight_mag=newlink.weight;
			else weight_mag=-newlink.weight;
			if (weight_mag>maxweight)
				maxweight=weight_mag;
		}
	}

    net.init(maxweight);
}

InnovLinkGene *InnovGenome::find_link(int in_node_id, int out_node_id, bool is_recurrent) {
    for(InnovLinkGene &g: links) {
        if( (g.in_node_id() == in_node_id)
            && (g.out_node_id() == out_node_id)
            && (g.is_recurrent() == is_recurrent) ) {

            return &g;
        }
    }

    return nullptr;
}

InnovNodeGene *InnovGenome::get_node(int id) {
    return node_lookup.find(id);
}

node_index_t InnovGenome::get_node_index(int id) {
    node_index_t i = get_node(id) - nodes.data();
    assert(nodes[i].node_id == id);
    return i;
}

void InnovGenome::delete_if_orphaned_hidden_node(int node_id) {
    InnovNodeGene *node = get_node(node_id);
    if(node->type != nodetype::HIDDEN)
        return;

    bool found_link;
    for(InnovLinkGene &link: links) {
        if(link.in_node_id() == node_id || link.out_node_id() == node_id) {
            found_link = true;
            break;
        }
    }

    if(!found_link) {
        auto iterator = nodes.begin() + (node - nodes.data());
        assert(iterator->node_id == node_id);
        nodes.erase(iterator);
    }
}

void InnovGenome::delete_link(InnovLinkGene *link) {
    auto iterator = find_if(links.begin(), links.end(),
                            [link](const InnovLinkGene &l) {
                                return l.innovation_num == link->innovation_num;
                            });
    assert(iterator != links.end());
    links.erase(iterator);
}
