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
#include "genome.h"

#include <assert.h>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <map> //todo: remove after innovations working
#include <sstream>

using namespace NEAT;
using namespace std;

class RecurrencyChecker {
private:
    size_t nnodes;
    LinkGene **links;
    size_t nlinks;

    static bool cmp_sort(const LinkGene *x, const LinkGene *y) {
        return x->out_node_id() < y->out_node_id();
    }

    static bool cmp_find(const LinkGene *x, int node_id) {
        return x->out_node_id() < node_id;
    }

    bool find(int node_id, LinkGene ***curr) {
        if(*curr == nullptr) {
            auto it = std::lower_bound(links, links + nlinks, node_id, cmp_find);
            if(it == links + nlinks) return false;
            if((*it)->out_node_id() != node_id) return false;

            *curr = it;
            return true;
        } else {
            (*curr)++;
            if(*curr >= (links + nlinks)) return false;
            if((**curr)->out_node_id() != node_id) return false;
            return true;
        }
    }

    // This checks a POTENTIAL link between a potential in_node and potential out_node to see if it must be recurrent 
    bool is_recur(int in_id, int out_id, int &count, int thresh) {
        ++count;  //Count the node as visited
        if(count > thresh) {
            return false;  //Short out the whole thing- loop detected
        }

        if (in_id==out_id) return true;
        else {
            LinkGene **gene = nullptr;
            while( find(in_id, &gene) ) {
                //But skip links that are already recurrent
                //(We want to check back through the forward flow of signals only
                if(!(*gene)->is_recurrent()) {
                    if( is_recur((*gene)->in_node_id(), out_id, count, thresh) )
                        return true;
                }
            }
            return false;
        }
    }

public:
    RecurrencyChecker(size_t nnodes_,
                      vector<LinkGene> &genome_links,
                      LinkGene **buf_links) {
        nnodes = nnodes_;
        links = buf_links;

        nlinks = 0;
        for(size_t i = 0; i < genome_links.size(); i++) {
            LinkGene *g = &genome_links[i];
            if(g->enable) {
                links[nlinks++] = g;
            }
        }
        std::sort(links, links + nlinks, cmp_sort);
    }

    bool is_recur(int in_node_id, int out_node_id) {
        //These are used to avoid getting stuck in an infinite loop checking
        //for recursion
        //Note that we check for recursion to control the frequency of
        //adding recurrent links rather than to prevent any paricular
        //kind of error
        int thresh=nnodes*nnodes;
        int count = 0;

        if(is_recur(in_node_id, out_node_id, count, thresh)) {
            return true;
        }

        //ADDED: CONSIDER connections out of outputs recurrent
        //todo: this was fixed to use place instead of type,
        //      but not clear if this logic is desirable. Shouldn't it
        //      just be checking if the output node is OUTPUT?
        /*
          if (((in_node->place)==OUTPUT)||
          ((out_node->place)==OUTPUT))
          return true;
        */
        return false;
    }
    
};

void Genome::reset(int new_id) {
    genome_id = new_id;
    traits.clear();
    nodes.clear();
    links.clear();
}

Genome::Genome()
    : node_lookup(nodes) {
}

Genome::Genome(int id, const vector<Trait> &t, const vector<NodeGene> &n, const vector<LinkGene> &g)
    : node_lookup(nodes) {
	genome_id=id;
	traits=t;
    links = g;
    nodes = n;
}

Genome::Genome(int id, std::ifstream &iFile)
    : node_lookup(nodes) {

    load_from_file(id, iFile);
}

Genome* Genome::new_Genome_load(char *filename) {
	Genome *newgenome;
	int id;

	char curword[20];  //max word size of 20 characters
	std::ifstream iFile(filename);
	iFile>>curword;

	//Bypass initial comment
	if (strcmp(curword,"/*")==0) {
		iFile>>curword;
		while (strcmp(curword,"*/")!=0) {
			printf("%s ",curword);
			iFile>>curword;
		}
		iFile>>curword;
	}
	iFile>>id;
	newgenome=new Genome(id,iFile);
	iFile.close();

	return newgenome;
}

Genome::~Genome() {
}

void Genome::verify() {
#ifdef NDEBUG
    return;
#else

	//Check for NodeGenes being out of order
    for(size_t i = 1, n = nodes.size(); i < n; i++) {
        assert( nodes[i-1].node_id < nodes[i].node_id );
    }

    {
        //Check links reference valid nodes.
        for(LinkGene &gene: links) {
            assert( get_node(gene.in_node_id()) );
            assert( get_node(gene.out_node_id()) );
        }
    }

	//Make sure there are no duplicate genes
	for(LinkGene &gene: links) {
		for(LinkGene &gene2: links) {
            if(&gene != &gene2) {
                assert( (gene.is_recurrent() != gene2.is_recurrent())
                        || (gene2.in_node_id() != gene.in_node_id())
                        || (gene2.out_node_id() != gene.out_node_id()) );
            }
		}
	}
#endif
}

void Genome::print_to_file(std::ostream &outFile) {
    outFile<<"genomestart "<<genome_id<<std::endl;

	//Output the traits
    for(auto &t: traits)
        t.print_to_file(outFile);

    //Output the nodes
    for(auto &n: nodes)
        n.print_to_file(outFile);

    //Output the genes
    for(auto &g: links)
        g.print_to_file(outFile);

    outFile << "genomeend " << genome_id << std::endl;
}

void Genome::load_from_file(int id, std::istream &iFile) {
	char curword[128];  //max word size of 128 characters
	char curline[1024]; //max line size of 1024 characters
	char delimiters[] = " \n";

	int done=0;

	//int pause;

	genome_id=id;

	iFile.getline(curline, sizeof(curline));
	int wordcount = NEAT::getUnitCount(curline, delimiters);
	int curwordnum = 0;

	//Loop until file is finished, parsing each line
	while (!done) {

        //std::cout << curline << std::endl;

		if (curwordnum > wordcount || wordcount == 0) {
			iFile.getline(curline, sizeof(curline));
			wordcount = NEAT::getUnitCount(curline, delimiters);
			curwordnum = 0;
		}
        
        std::stringstream ss(curline);
		//strcpy(curword, NEAT::getUnit(curline, curwordnum++, delimiters));
        ss >> curword;

		//printf(curword);
		//printf(" test\n");
		//Check for end of Genome
		if (strcmp(curword,"genomeend")==0) {
			//strcpy(curword, NEAT::getUnit(curline, curwordnum++, delimiters));
            ss >> curword;
			int idcheck = atoi(curword);
			//iFile>>idcheck;
			if (idcheck!=genome_id) printf("ERROR: id mismatch in genome");
			done=1;
		}

		//Ignore genomestart if it hasn't been gobbled yet
		else if (strcmp(curword,"genomestart")==0) {
			++curwordnum;
			//cout<<"genomestart"<<endl;
		}

		//Ignore comments surrounded by - they get printed to screen
		else if (strcmp(curword,"/*")==0) {
			//strcpy(curword, NEAT::getUnit(curline, curwordnum++, delimiters));
            ss >> curword;
			while (strcmp(curword,"*/")!=0) {
				//cout<<curword<<" ";
				//strcpy(curword, NEAT::getUnit(curline, curwordnum++, delimiters));
                ss >> curword;
			}
			//cout<<endl;
		}

		//Read in a trait
		else if (strcmp(curword,"trait")==0) {
			Trait *newtrait;

			char argline[1024];
			//strcpy(argline, NEAT::getUnits(curline, curwordnum, wordcount, delimiters));

			curwordnum = wordcount + 1;

            ss.getline(argline, 1024);
			//Allocate the new trait
			newtrait=new Trait(argline);

			//Add trait to vector of traits
			traits.push_back(newtrait);
		}

		//Read in a node
		else if (strcmp(curword,"node")==0) {
			char argline[1024];
			curwordnum = wordcount + 1;
            
            ss.getline(argline, 1024);
			nodes.emplace_back(argline);
		}

		//Read in a LinkGene
		else if (strcmp(curword,"gene")==0) {
			char argline[1024];
			curwordnum = wordcount + 1;

            ss.getline(argline, 1024);
			links.emplace_back(argline);
		}

	}
}

int Genome::get_last_node_id() {
    return nodes.back().node_id + 1;
}

double Genome::get_last_gene_innovnum() {
    return links.back().innovation_num + 1;
}

void Genome::duplicate_into(Genome &offspring, int new_id) {
    offspring.genome_id = new_id;
    offspring.traits = traits;
    offspring.links = links;
    offspring.nodes = nodes;
}

void Genome::mutate_random_trait() {
    rng.element(traits).mutate(rng);
}

void Genome::mutate_link_trait(int times) {
    for(int i = 0; i < times; i++) {
        int trait_id = 1 + rng.index(traits);
        LinkGene &gene = rng.element(links);
        
        if(!gene.frozen) {
            gene.set_trait_id(trait_id);
        }
    }
}

void Genome::mutate_node_trait(int times) {
    for(int i = 0; i < times; i++) {
        int trait_id = 1 + rng.index(traits);
        NodeGene &node = rng.element(nodes);

        if(!node.frozen) {
            node.set_trait_id(trait_id);
        }
    }

    //TRACK INNOVATION! - possible future use
    //for any gene involving the mutated node, perturb that gene's
    //mutation number
}

void Genome::mutate_link_weights(double power,double rate,mutator mut_type) {
	//Go through all the LinkGenes and perturb their link's weights

	double num = 0.0; //counts gene placement
	double gene_total = (double)links.size();
	double endpart = gene_total*0.8; //Signifies the last part of the genome
	double powermod = 1.0; //Modified power by gene number
	//The power of mutation will rise farther into the genome
	//on the theory that the older genes are more fit since
	//they have stood the test of time

	bool severe = rng.prob() > 0.5;  //Once in a while really shake things up

	//Loop on all links  (ORIGINAL METHOD)
	for(LinkGene &gene: links) {

		//The following if determines the probabilities of doing cold gaussian
		//mutation, meaning the probability of replacing a link weight with
		//another, entirely random weight.  It is meant to bias such mutations
		//to the tail of a genome, because that is where less time-tested links
		//reside.  The gausspoint and coldgausspoint represent values above
		//which a random float will signify that kind of mutation.  

		//Don't mutate weights of frozen links
		if (!(gene.frozen)) {
            double gausspoint;
            double coldgausspoint;

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
			double randnum = rng.posneg()*rng.prob()*power*powermod;
			if (mut_type==GAUSSIAN) {
				double randchoice = rng.prob();
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

void Genome::mutate_toggle_enable(int times) {
    for(int i = 0; i < times; i++) {
        LinkGene &gene = rng.element(links);

        if(!gene.enable) {
            gene.enable = true;
        } else {
			//We need to make sure that another gene connects out of the in-node
			//Because if not a section of network will break off and become isolated
            bool found = false;
            for(LinkGene &checkgene: links) {
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

void Genome::mutate_gene_reenable() {
	//Search for a disabled gene
    for(LinkGene &g: links) {
        if(!g.enable) {
            g.enable = true;
            break;
        }
    }
}

//todo: delete debug code
namespace NEAT {
    int *__cur_node_id;
    double *__cur_innov_num;

    static map<InnovationId, vector<IndividualInnovation>> id2inds;

    void reset_debug() {
        id2inds.clear();
    }

    void apply_debug() {
        vector<IndividualInnovation> masters;
        for(auto &kv: id2inds) {
            auto &inds = kv.second;
            auto &master = inds.front();
            masters.push_back(master);
        }        

        std::sort(masters.begin(), masters.end(),
                  [] (const IndividualInnovation &x, const IndividualInnovation &y) {
                      return x.population_index < y.population_index;
                  });
        

        for(auto &master: masters) {
            auto &inds = id2inds[master.id];

            Innovation *innov;
            switch(master.id.innovation_type) {
            case NEWNODE: {
                innov = new Innovation(master.id,
                                       master.parms,
                                       *__cur_innov_num,
                                       *__cur_innov_num + 1,
                                       *__cur_node_id);
                *__cur_innov_num += 2;
                *__cur_node_id += 1;
            } break;
            case NEWLINK: {
                innov = new Innovation(master.id,
                                           master.parms,
                                           *__cur_innov_num);
                    *__cur_innov_num += 1.0;
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


bool Genome::mutate_add_node(int population_index,
                             vector<Innovation*> &innovs,
                             int &curnode_id,
                             double &curinnov) {
    LinkGene *splitlink = nullptr;
    {
        for(int i = 0; !splitlink && i < 20; i++) {
            LinkGene &g = rng.element(links);
            //If either the link is disabled, or it has a bias input, try again
            if( g.enable && get_node(g.in_node_id())->place != BIAS ) {
                splitlink = &g;
            }
        }
        //We couldn't find anything, so say goodbye!
        if (!splitlink) {
            return false;
        }
    }

    splitlink->enable = false;

    auto create_genes = [this, splitlink] (const Innovation *innov) {

        NodeGene newnode(NEURON, innov->newnode_id, HIDDEN);

        LinkGene newlink1(splitlink->trait_id(),
                          1.0,
                          innov->id.node_in_id,
                          innov->newnode_id,
                          splitlink->is_recurrent(),
                          innov->innovation_num1,
                          0);

        LinkGene newlink2(splitlink->trait_id(),
                          splitlink->weight(),
                          innov->newnode_id,
                          innov->id.node_out_id,
                          false,
                          innov->innovation_num2,
                          0);    

        add_link(this->links, newlink1);
        add_link(this->links, newlink2);
        add_node(this->nodes, newnode);
    };

    InnovationId innov_id(splitlink->in_node_id(),
                          splitlink->out_node_id(),
                          splitlink->innovation_num);
    InnovationParms innov_parms;

    id2inds[innov_id].emplace_back(population_index, innov_id, innov_parms, create_genes);

	return true;

}

bool Genome::mutate_add_link(int population_index,
                             vector<Innovation*> &innovs,
                             double &curinnov,
                             int tries) {
    LinkGene *recur_checker_buf[links.size()];
    RecurrencyChecker recur_checker(nodes.size(), links, recur_checker_buf);

	NodeGene *in_node = nullptr; //Pointers to the nodes
	NodeGene *out_node = nullptr; //Pointers to the nodes

	//Decide whether to make this recurrent
	bool do_recur = rng.prob() < NEAT::recur_only_prob;

    // Try to find nodes for link.
    {
        bool found_nodes = false;

        //Find the first non-sensor so that the to-node won't look at sensors as
        //possible destinations
        int first_nonsensor = 0;
        for(; nodes[first_nonsensor].get_type() == SENSOR; first_nonsensor++) {
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

            found_nodes = !link_exists(in_node->node_id, out_node->node_id, do_recur)
                                 && (do_recur == recur_checker.is_recur(in_node->node_id, out_node->node_id));
        }

        assert( out_node->type != SENSOR );

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
        double newweight = rng.posneg() * rng.prob() * 1.0;

        InnovationParms innov_parms(newweight, trait_id);

        auto create_genes = [this] (const Innovation *innov) {

            LinkGene newlink(innov->parms.new_trait_id,
                             innov->parms.new_weight,
                             innov->id.node_in_id,
                             innov->id.node_out_id,
                             innov->id.recur_flag,
                             innov->innovation_num1,
                             innov->parms.new_weight);

            add_link(this->links, newlink);
        };

        id2inds[innov_id].emplace_back(population_index, innov_id, innov_parms, create_genes);
    }

    return true;
}

//Adds a new gene that has been created through a mutation in the
//*correct order* into the list of links in the genome
void Genome::add_link(vector<LinkGene> &glist, const LinkGene &g) {
  vector<LinkGene>::iterator curgene;
  double inum=g.innovation_num;

  curgene=glist.begin();
  while ((curgene!=glist.end())&&
	 ((curgene->innovation_num)<inum)) {
    ++curgene;
  }

  glist.insert(curgene, g);
}

//todo: use binary search.
void Genome::add_node(vector<NodeGene> &nlist, const NodeGene &n) {
	vector<NodeGene>::iterator curnode;

	int id=n.node_id;

	curnode=nlist.begin();
	while ((curnode!=nlist.end())&&
		((curnode->node_id)<id)) 
		++curnode;

	nlist.insert(curnode, n);

}

void Genome::mate_multipoint(Genome *g,
                             Genome *offspring,
                             int genomeid,
                             double fitness1,
                             double fitness2) {

    vector<LinkGene> &links1 = this->links;
    vector<LinkGene> &links2 = g->links;

	//The baby Genome will contain these new Traits, NodeGenes, and LinkGenes
    offspring->reset(genomeid);
	vector<Trait> &newtraits = offspring->traits;
	vector<NodeGene> &newnodes = offspring->nodes;   
	vector<LinkGene> &newlinks = offspring->links;    

	vector<LinkGene>::iterator curgene2;  //Checks for link duplication

	//iterators for moving through the two parents' traits
	vector<Trait*>::iterator p1trait;
	vector<Trait*>::iterator p2trait;

	//iterators for moving through the two parents' links
	vector<LinkGene>::iterator p1gene;
	vector<LinkGene>::iterator p2gene;
	double p1innov;  //Innovation numbers for links inside parents' Genomes
	double p2innov;
	vector<NodeGene>::iterator curnode;  //For checking if NodeGenes exist already 

	bool disable;  //Set to true if we want to disabled a chosen gene

	disable=false;
	LinkGene newgene;

	bool p1better; //Tells if the first genome (this one) has better fitness or not

	bool skip;

	//First, average the Traits from the 2 parents to form the baby's Traits
	//It is assumed that trait lists are the same length
	//In the future, may decide on a different method for trait mating
    assert(traits.size() == g->traits.size());
    for(size_t i = 0, n = traits.size(); i < n; i++) {
        newtraits.emplace_back(traits[i], g->traits[i]);
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
    for(NodeGene &node: g->nodes) {
		if( (node.place == INPUT)
            || (node.place == BIAS)
            || (node.place == OUTPUT)) {

            //Add the new node
            add_node(newnodes, node);
        }
    }

	//Now move through the LinkGenes of each parent until both genomes end
    Genome *genome1 = this;
    Genome *genome2 = g;
	p1gene=links1.begin();
	p2gene=(links2).begin();
	while( !((p1gene==links1.end()) && (p2gene==(links2).end())) ) {
        ProtoLinkGene protogene;

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
            NodeGene new_inode;
            NodeGene new_onode;

            //Next check for the nodes, add them if not in the baby Genome already
            NodeGene *inode = protogene.in();
            NodeGene *onode = protogene.out();

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

            } //End NodeGene checking section- NodeGenes are now in new Genome

            //Add the LinkGene
            newgene = LinkGene(protogene.gene(),
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

void Genome::mate_multipoint_avg(Genome *g,
                                 Genome *offspring,
                                 int genomeid,
                                 double fitness1,
                                 double fitness2) {
    vector<LinkGene> &links1 = this->links;
    vector<LinkGene> &links2 = g->links;

	//The baby Genome will contain these new Traits, NodeGenes, and LinkGenes
    offspring->reset(genomeid);
	vector<Trait> &newtraits = offspring->traits;
	vector<NodeGene> &newnodes = offspring->nodes;
	vector<LinkGene> &newlinks = offspring->links;

	vector<LinkGene>::iterator curgene2; //Checking for link duplication

	//iterators for moving through the two parents' links
	vector<LinkGene>::iterator p1gene;
	vector<LinkGene>::iterator p2gene;
	double p1innov;  //Innovation numbers for links inside parents' Genomes
	double p2innov;
	vector<NodeGene>::iterator curnode;  //For checking if NodeGenes exist already 

	//This LinkGene is used to hold the average of the two links to be averaged
	LinkGene avgene(0,0,0,0,0,0,0);
	LinkGene newgene;

	bool skip;

	bool p1better;  //Designate the better genome

	//First, average the Traits from the 2 parents to form the baby's Traits
	//It is assumed that trait lists are the same length
	//In future, could be done differently
    for(size_t i = 0, n = traits.size(); i < n; i++) {
        newtraits.emplace_back(traits[i], g->traits[i]);
	}

	//NEW 3/17/03 Make sure all sensors and outputs are included
    for(NodeGene &node: g->nodes) {
		if (((node.place)==INPUT)||
			((node.place)==OUTPUT)||
			((node.place)==BIAS)) {

            add_node(newnodes, node);
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


	//Now move through the LinkGenes of each parent until both genomes end
    Genome *genome1 = this;
    Genome *genome2 = g;
	p1gene=links1.begin();
	p2gene=(links2).begin();
	while(!((p1gene==links1.end()) && (p2gene==(links2).end()))) {
        ProtoLinkGene protogene;

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

            //Next check for the nodes, add them if not in the baby Genome already
            NodeGene *inode = protogene.in();
            NodeGene *onode = protogene.out();

            //Check for inode in the newnodes list
            NodeGene new_inode;
            NodeGene new_onode;
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

            } //End NodeGene checking section- NodeGenes are now in new Genome

            //Add the LinkGene
            newgene = LinkGene(protogene.gene(),
                           protogene.gene()->trait_id(),
                           new_inode.node_id,
                           new_onode.node_id);

            newlinks.push_back(newgene);

        }  //End if which checked for link duplicationb

    }
}

double Genome::compatibility(Genome *g) {
    vector<LinkGene> &links1 = this->links;
    vector<LinkGene> &links2 = g->links;


	//Innovation numbers
	double p1innov;
	double p2innov;

	//Intermediate value
	double mut_diff;

	//Set up the counters
	double num_disjoint=0.0;
	double num_excess=0.0;
	double mut_diff_total=0.0;
	double num_matching=0.0;  //Used to normalize mutation_num differences

	//Now move through the LinkGenes of each potential parent 
	//until both Genomes end
	vector<LinkGene>::iterator p1gene = links1.begin();
	vector<LinkGene>::iterator p2gene = links2.begin();

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
		//difference between mutation_nums for any two matching LinkGenes
		//in the Genome

		//Normalizing for genome size
		//return (disjoint_coeff*(num_disjoint/max_genome_size)+
		//  excess_coeff*(num_excess/max_genome_size)+
		//  mutdiff_coeff*(mut_diff_total/num_matching));


		//Look at disjointedness and excess in the absolute (ignoring size)

		return (NEAT::disjoint_coeff*(num_disjoint/1.0)+
			NEAT::excess_coeff*(num_excess/1.0)+
			NEAT::mutdiff_coeff*(mut_diff_total/num_matching));
}

double Genome::trait_compare(Trait *t1,Trait *t2) {

	int id1=t1->trait_id;
	int id2=t2->trait_id;
	int count;
	double params_diff=0.0; //Measures parameter difference

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

int Genome::extrons() {
	int total=0;

    for(LinkGene &g: links) {
		if (!g.enable) ++total;
	}

	return total;
}

void Genome::randomize_traits() {
    for(NodeGene &node: nodes) {
		node.set_trait_id(1 + rng.index(traits));
	}

    for(LinkGene &gene: links) {
		gene.set_trait_id(1 + rng.index(traits));
	}
}

inline Trait &get_trait(vector<Trait> &traits, int trait_id) {
    Trait &t = traits[trait_id - 1];
    assert(t.trait_id == trait_id);
    return t;
}

Trait &Genome::get_trait(const NodeGene &node) {
    return ::get_trait(traits, node.get_trait_id());
}

Trait &Genome::get_trait(const LinkGene &gene) {
    return ::get_trait(traits, gene.trait_id());
}

bool Genome::link_exists(int in_node_id, int out_node_id, bool is_recurrent) {
    for(LinkGene &g: links) {
        if( (g.in_node_id() == in_node_id)
            && (g.out_node_id() == out_node_id)
            && (g.is_recurrent() == is_recurrent) ) {

            return true;
        }
    }

    return false;
}

NodeGene *Genome::get_node(int id) {
    return node_lookup.find(id);
}

void NEAT::print_Genome_tofile(Genome *g,const char *filename) {

    std::string file = "nero/data/neat/";
    file += filename;
	std::ofstream oFile(file.c_str());
	g->print_to_file(oFile);
	oFile.close();
}

