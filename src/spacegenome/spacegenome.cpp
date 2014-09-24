#include "spacegenome.h"

#include "util.h"
#include <climits>

using namespace NEAT;
using namespace std;

#define DIST_FACT 3 //todo: better name. put in env.

SpaceGenome::SpaceGenome(rng_t rng_,
                         size_t ntraits,
                         size_t ninputs,
                         size_t noutputs,
                         size_t nhidden)
    : SpaceGenome() {

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

    size_t index_bias;
    size_t index_inputs;
    size_t index_outputs;
    size_t index_hidden;
    {
        short x = -DIST_FACT;
        short y = (ninputs+1) / 2;

        //Bias node
        index_bias = nodes.size();
        add_node(SpaceNodeGene(nodetype::BIAS, {x, y--}));

        //Sensor nodes
        index_inputs = nodes.size();
        for(size_t i = 0; i < ninputs; i++) {
            add_node(SpaceNodeGene(nodetype::SENSOR, {x, y--}));
        }

        x = DIST_FACT;
        y = noutputs / 2;
        //Output nodes
        index_outputs = nodes.size();
        for(size_t i = 0; i < noutputs; i++) {
            add_node(SpaceNodeGene(nodetype::OUTPUT, {x, y--}));
        }

        x = 0;
        y = nhidden / 2;
        //Hidden nodes
        index_hidden = nodes.size();
        for(size_t i = 0; i < nhidden; i++) {
            add_node(SpaceNodeGene(nodetype::HIDDEN, {x, y--}));
        }
    }

    assert(nhidden > 0);

    auto create_link = [=] (size_t in_index, size_t out_index) {
        add_link( SpaceLinkGene(rng.element(traits).trait_id,
                                rng.prob(),
                                nodes[in_index].location,
                                nodes[out_index].location) );
    };

    //Create links from Bias to all hidden
    for(size_t i = 0; i < nhidden; i++) {
        create_link(index_bias, i + index_hidden);
    }

    //Create links from all inputs to all hidden
    for(size_t i = 0; i < ninputs; i++) {
        for(size_t j = 0; j < nhidden; j++) {
            create_link(i + index_inputs, j + index_hidden);
        }
    }

    //Create links from all hidden to all output
    for(size_t i = 0; i < nhidden; i++) {
        for(size_t j = 0; j < noutputs; j++) {
            create_link(i + index_hidden, j + index_outputs);
        }
    }    

}

void SpaceGenome::duplicate_into(SpaceGenome *offspring) {
    offspring->traits = traits;
    offspring->links = links;
    offspring->nodes = nodes;
}

SpaceGenome &SpaceGenome::operator=(const SpaceGenome &other) {
    rng = other.rng;
    genome_id = other.genome_id;
    traits = other.traits;
    nodes = other.nodes;
    links = other.links;
    return *this;
}

unique_ptr<Genome> SpaceGenome::make_default() const {
    return unique_ptr<Genome>(new SpaceGenome());
}

unique_ptr<Genome> SpaceGenome::make_clone() const {
    SpaceGenome *g = new SpaceGenome();
    *g = *this;
    return unique_ptr<Genome>(g);
}

void SpaceGenome::print(std::ostream &out) {
    out<<"spacegenomestart "<<genome_id<<std::endl;

	//Output the traits
    for(auto &t: traits)
        t.print_to_file(out);

    //Output the nodes
    for(auto &n: nodes)
        out << n << endl;

    //Output the genes
    for(auto &l: links)
        out << l << endl;

    out << "spacegenomeend " << genome_id << std::endl;
}

void SpaceGenome::verify() {
#ifdef NDEBUG
    return;
#else
    return;
#endif
}

Genome::Stats SpaceGenome::get_stats() {
    return {nodes.size(), links.size()};
}

void SpaceGenome::init_phenotype(Network &net) {
	real_t maxweight=0.0; //Compute the maximum weight for adaptation purposes
	real_t weight_mag; //Measures absolute value of weights

    net.reset();
    vector<NNode> &netnodes = net.nodes;

	//Create the nodes
	for(SpaceNodeGene &node: nodes) {
        netnodes.emplace_back(node.type);
	}

	//Create the links by iterating through the genes
    for(SpaceLinkGene &gene: links) {
        node_index_t inode = get_node_index(gene.in_node_loc);
        node_index_t onode = get_node_index(gene.out_node_loc);

		//NOTE: This line could be run through a recurrency check if desired
		// (no need to in the current implementation of NEAT)
		netnodes[onode].incoming.emplace_back(gene.weight, inode);

        Link &newlink = netnodes[onode].incoming.back();

		//Keep track of maximum weight
		if (newlink.weight>0)
			weight_mag=newlink.weight;
		else weight_mag=-newlink.weight;
		if (weight_mag>maxweight)
			maxweight=weight_mag;
	}

    net.init(maxweight);
}

void SpaceGenome::mutate() {
    rng_t::prob_switch_t op = rng.prob_switch();

    if( op.prob_case(NEAT::mutate_add_node_prob) ) {
        mutate_add_node();
    } else if( op.prob_case(NEAT::mutate_add_link_prob) ) {
        mutate_add_link();
    } else if( op.prob_case(NEAT::mutate_delete_link_prob) ) {
        mutate_delete_link();
    } else if( op.prob_case(NEAT::mutate_delete_node_prob) ) {
        mutate_delete_node();
    } else {
        //Only do other mutations when not doing sturctural mutations
        if( rng.under(NEAT::mutate_random_trait_prob) ) {
            mutate_random_trait();
        }
        if( rng.under(NEAT::mutate_link_trait_prob) ) {
            mutate_link_trait();
        }
        if( rng.under(NEAT::mutate_node_trait_prob) ) {
            mutate_node_trait();
        }
        if( rng.under(NEAT::mutate_link_weights_prob) ) {
            mutate_link_weights(NEAT::weight_mut_power, 1.0, GAUSSIAN);
        }
    }
}

void SpaceGenome::randomize_traits() {
    for(SpaceNodeGene &node: nodes) {
		node.trait_id = 1 + rng.index(traits);
	}
    for(SpaceLinkGene &gene: links) {
		gene.trait_id = 1 + rng.index(traits);
	}
}

void SpaceGenome::mutate_link_weights(real_t power,
                                      real_t rate,
                                      mutator mut_type) {
	//Go through all the SpaceLinkGenes and perturb their link's weights

	real_t num = 0.0; //counts gene placement
	real_t gene_total = (real_t)links.size();
	real_t endpart = gene_total*0.8; //Signifies the last part of the genome
	real_t powermod = 1.0; //Modified power by gene number
	//The power of mutation will rise farther into the genome
	//on the theory that the older genes are more fit since
	//they have stood the test of time

	bool severe = rng.prob() > 0.5;  //Once in a while really shake things up

	//Loop on all links  (ORIGINAL METHOD)
	for(SpaceLinkGene &gene: links) {

		//The following if determines the probabilities of doing cold gaussian
		//mutation, meaning the probability of replacing a link weight with
		//another, entirely random weight.  It is meant to bias such mutations
		//to the tail of a genome, because that is where less time-tested links
		//reside.  The gausspoint and coldgausspoint represent values above
		//which a random float will signify that kind of mutation.  

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
                gene.weight+=randnum;
            else if (randchoice > coldgausspoint)
                gene.weight=randnum;
        }
        else if (mut_type==COLDGAUSSIAN)
            gene.weight=randnum;

        //Cap the weights at 8.0 (experimental)
        if (gene.weight > 8.0) gene.weight = 8.0;
        else if (gene.weight < -8.0) gene.weight = -8.0;

        num+=1.0;
	} //end for loop
}

void SpaceGenome::mutate_add_link() {
	SpaceNodeGene *in_node = nullptr; //Pointers to the nodes
	SpaceNodeGene *out_node = nullptr; //Pointers to the nodes

    // Try to find nodes for link.
    for(int attempt = 0; (in_node == nullptr) && attempt < 20; attempt++) {
        //Find the first non-sensor so that the to-node won't look at sensors as
        //possible destinations
        
        //todo: nodelookup could do this with a binary search.
        int first_nonsensor = 0;
        for(; is_input(nodes[first_nonsensor].type); first_nonsensor++) {
        }

        out_node = &rng.element(nodes, first_nonsensor);

        NodeLocation desired_location;
        create_random_node_location(out_node->location,
                                    desired_location,
                                    false);

        double closest_dist;
        for(SpaceNodeGene &node_: nodes) {
            SpaceNodeGene *node = &node_;
            if(node == out_node)
                continue;

            double dist = node->location.dist(desired_location);
            if( (in_node == nullptr)
                || (dist < closest_dist)
                || ((dist == closest_dist) && rng.boolean()) ) {

                if( find_link(out_node->location, node->location) == nullptr ) {
                    in_node = node;
                    closest_dist = dist;
                }
            }
        }
    }

    //Continue only if an open link was found
    if(in_node == nullptr) {
        return;
    }
    assert( !is_input(out_node->type) );

    // Create the gene.
    int trait_id = 1 + rng.index(traits);
    real_t weight = rng.posneg() * rng.prob() * 1.0;
    SpaceLinkGene link{trait_id, weight, in_node->location, out_node->location};

    add_link(link);
}

void SpaceGenome::mate(SpaceGenome *genome1,
                       SpaceGenome *genome2,
                       SpaceGenome *offspring,
                       real_t fitness1,
                       real_t fitness2) {
    if(fitness2 > fitness1) {
        std::swap(genome1, genome2);
        std::swap(fitness1, fitness2);
    }

    mate_singlepoint(genome1, genome2, offspring);

    if( !offspring->rng.under(NEAT::mate_only_prob) ||
        (genome2->genome_id == genome1->genome_id) ) {

        offspring->mutate();
    }

}

void SpaceGenome::reset() {
    traits.clear();
    nodes.clear();
    links.clear();
}

void SpaceGenome::mutate_random_trait() {
    rng.element(traits).mutate(rng);
}

void SpaceGenome::mutate_link_trait(int times) {
    for(int i = 0; i < times; i++) {
        int trait_id = 1 + rng.index(traits);
        SpaceLinkGene &gene = rng.element(links);
        
        gene.trait_id = trait_id;
    }
}

void SpaceGenome::mutate_node_trait(int times) {
    for(int i = 0; i < times; i++) {
        int trait_id = 1 + rng.index(traits);
        SpaceNodeGene &node = rng.element(nodes);

        node.trait_id = trait_id;
    }
}

void SpaceGenome::mutate_add_node() {
    SpaceLinkGene *splitlink = nullptr;
    {
        for(int i = 0; !splitlink && i < 20; i++) {
            SpaceLinkGene &g = rng.element(links);
            //If link has a bias input, try again
            if( get_node(g.in_node_loc)->type != nodetype::BIAS ) {
                splitlink = &g;
            }
        }
        //We couldn't find anything, so say goodbye!
        if (!splitlink) {
            return;
        }
    }

    SpaceNodeGene *in_node = get_node(splitlink->in_node_loc);
    SpaceNodeGene *out_node = get_node(splitlink->out_node_loc);
    NodeLocation newnode_location;
    if(!create_random_node_location(in_node->location,
                                    out_node->location,
                                    newnode_location)) {
        return;
    }

    SpaceNodeGene newnode(nodetype::HIDDEN, newnode_location);

    SpaceLinkGene newlink1(splitlink->trait_id,
                           1.0,
                           splitlink->in_node_loc,
                           newnode.location);
    SpaceLinkGene newlink2(splitlink->trait_id,
                           splitlink->weight,
                           newnode_location,
                           splitlink->out_node_loc);

    delete_link(*splitlink);
    add_link(newlink1);
    add_link(newlink2);
    add_node(newnode);
}

void SpaceGenome::mutate_delete_node() {
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
    SpaceNodeGene node = nodes[node_index];
    assert(node.type == nodetype::HIDDEN);

    nodes.erase(nodes.begin() + node_index);

    //todo: we should have a way to look up links by in/out id
    erase_if(links,
             [&node] (const SpaceLinkGene &link) {
                 return link.in_node_loc == node.location
                     || link.out_node_loc == node.location;
             });
}

void SpaceGenome::mutate_delete_link() {
    if(links.size() <= 1)
        return;

    size_t link_index = rng.index(links);
    SpaceLinkGene link = links[link_index];
    links.erase(links.begin() + link_index);

    delete_if_orphaned_hidden_node(link.in_node_loc);
    delete_if_orphaned_hidden_node(link.out_node_loc);
}

// Note: genome1 should be fitter than genome2
void SpaceGenome::mate_singlepoint(SpaceGenome *genome1,
                                   SpaceGenome *genome2,
                                   SpaceGenome *offspring) {
    

    rng_t &rng = offspring->rng;
    vector<SpaceLinkGene> &links1 = genome1->links;
    vector<SpaceLinkGene> &links2 = genome2->links;

	//The baby SpaceGenome will contain these new Traits, SpaceNodeGenes, and SpaceLinkGenes
    offspring->reset();

	//First, average the Traits from the 2 parents to form the baby's Traits
	//It is assumed that trait lists are the same length
	//In the future, may decide on a different method for trait mating
    assert(genome1->traits.size() == genome2->traits.size());
    for(size_t i = 0, n = genome1->traits.size(); i < n; i++) {
        offspring->traits.emplace_back(genome1->traits[i], genome2->traits[i]);
    }

	//Make sure all sensors and outputs are included
    for(SpaceNodeGene &node: genome1->nodes) {
		if(node.type != nodetype::HIDDEN) {
            //Add the new node
            offspring->add_node(node);
        } else {
            break;
        }
    }

    auto add_node = [=] (const NodeLocation &loc) {
        if(offspring->get_node(loc))
            return;

        SpaceNodeGene *node = genome1->get_node(loc);
        if(node) {
            offspring->add_node(*node);
        } else {
            node = genome2->get_node(loc);
            assert(node != nullptr);
            offspring->add_node(*node);
        }
    };

    auto add_links = [=] (vector<SpaceLinkGene> &source,
                          size_t start, size_t end) {
        for(size_t i = start; i < end; i++) {
            SpaceLinkGene &link = source[i];
            add_node(link.in_node_loc);
            add_node(link.out_node_loc);

            SpaceLinkGene *existing_link = offspring->find_link(link.in_node_loc,
                                                                link.out_node_loc);
            if(existing_link == nullptr) {
                offspring->add_link(link);
            }
        }
    };

    size_t crossover_point;
    {
        size_t minlen = std::min(links1.size(), links2.size());
        crossover_point = rng.integer( int(0.25 * minlen),
                                       int(0.75 * minlen) );
    }
    if(rng.boolean()) {
        add_links(links1, 0, crossover_point);
        add_links(links2, crossover_point, links2.size());
    } else {
        add_links(links1, crossover_point, links1.size());
        add_links(links2, 0, crossover_point);
    }
}

void SpaceGenome::delete_if_orphaned_hidden_node(const NodeLocation &loc) {
    SpaceNodeGene *node = get_node(loc);
    if(node->type != nodetype::HIDDEN)
        return;

    bool found_link;
    for(SpaceLinkGene &link: links) {
        if(link.in_node_loc == loc || link.out_node_loc == loc) {
            found_link = true;
            break;
        }
    }

    if(!found_link) {
        auto iterator = nodes.begin() + (node - nodes.data());
        assert(iterator->location == loc);
        nodes.erase(iterator);
    }
}

void SpaceGenome::add_link(const SpaceLinkGene &l) {
    auto it = std::upper_bound(links.begin(), links.end(), l);
    links.insert(it, l);
}

void SpaceGenome::delete_link(const SpaceLinkGene &link) {
    auto it = std::lower_bound(links.begin(), links.end(), link);
    assert(it != links.end());
    assert(it->in_node_loc == link.in_node_loc);
    assert(it->out_node_loc == link.out_node_loc);
    links.erase(it);
}

SpaceLinkGene *SpaceGenome::find_link(const NodeLocation &in_node_loc,
                                      const NodeLocation &out_node_loc) {
    SpaceLinkGene key = SpaceLinkGene::create_search_key(in_node_loc, out_node_loc);
    auto it = std::lower_bound(links.begin(), links.end(), key);
    if(it == links.end())
        return nullptr;

    SpaceLinkGene *result = &(*it);
    if( (result->in_node_loc != in_node_loc)
        || (result->out_node_loc == out_node_loc) ) {
        return nullptr;
    }
    return result;
}

void SpaceGenome::add_node(const SpaceNodeGene &n) {
    auto it = std::upper_bound(nodes.begin(), nodes.end(), n);
    nodes.insert(it, n);
}

SpaceNodeGene *SpaceGenome::get_node(const NodeLocation &location) {
    return node_lookup.find(location);
}

node_index_t SpaceGenome::get_node_index(const NodeLocation &location) {
    node_index_t i = get_node(location) - nodes.data();
    assert(nodes[i].location == location);
    return i;
}

bool SpaceGenome::create_random_node_location(const NodeLocation &loc1,
                                              const NodeLocation &loc2,
                                              NodeLocation &result,
                                              bool empty_space_required,
                                              int maxtries) {
    NodeLocation center( std::round((loc1.x + loc2.x) / 2.0),
                         std::round((loc1.y + loc2.y) / 2.0) );
    return create_random_node_location(center, result, empty_space_required, maxtries);
}

bool SpaceGenome::create_random_node_location(const NodeLocation &center,
                                              NodeLocation &result,
                                              bool empty_space_required,
                                              int maxtries) {
    assert( (maxtries > 0) || (maxtries == -1) );

    for(int i = 0; (maxtries == -1) || (i < 10); i++) {
        int x = std::round(center.x + DIST_FACT*rng.gauss());
        int y = std::round(center.y + DIST_FACT*rng.gauss());
        
        if( (x < SHRT_MIN) || (x > SHRT_MAX) ) {
            warn("Exceeded short limits search for new node location: " << x);
            continue;
        } else if( (y < SHRT_MIN) || (y > SHRT_MAX) ) {
            warn("Exceeded short limits search for new node location: " << y);
            continue;
        }

        NodeLocation candidate{ short(x), short(y) };
        if(!empty_space_required) {
            result = candidate;
            return true;
        } else {
            SpaceNodeGene *existing = get_node(candidate);
            if(existing == nullptr) {
                result = candidate;
                return true;
            }
        }
    }

    return false;
}


//$$$
/*
#include "protoinnovlinkgene.h"
#include "recurrencychecker.h"
#include "util.h"
#include <assert.h>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <sstream>

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
        add_node(nodes, InnovNodeGene(SENSOR, node_id++, BIAS));

        //Sensor nodes
        for(size_t i = 0; i < ninputs; i++) {
            add_node(nodes, InnovNodeGene(SENSOR, node_id++, INPUT));
        }

        //Output nodes
        for(size_t i = 0; i < noutputs; i++) {
            add_node(nodes, InnovNodeGene(NEURON, node_id++, OUTPUT));
        }

        //Hidden nodes
        for(size_t i = 0; i < nhidden; i++) {
            add_node(nodes, InnovNodeGene(NEURON, node_id++, HIDDEN));
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

InnovGenome::InnovGenome(int id,
                         const vector<Trait> &t,
                         const vector<InnovNodeGene> &n,
                         const vector<InnovLinkGene> &g)
    : node_lookup(nodes) {
	genome_id=id;
	traits=t;
    links = g;
    nodes = n;
}

InnovGenome::~InnovGenome() {
}

int InnovGenome::get_last_node_id() {
    return nodes.back().node_id + 1;
}

real_t InnovGenome::get_last_gene_innovnum() {
    return links.back().innovation_num + 1;
}

void InnovGenome::mutate(CreateInnovationFunc create_innov) {
    //Do the mutation depending on probabilities of 
    //various mutations
    rng_t::prob_switch_t op = rng.prob_switch();

    if( op.prob_case(NEAT::mutate_add_node_prob) ) {
        mutate_add_node(create_innov);
    } else if( op.prob_case(NEAT::mutate_add_link_prob) ) {
        mutate_add_link(create_innov,
                        NEAT::newlink_tries);
    } else if( op.prob_case(NEAT::mutate_delete_link_prob) ) {
        mutate_delete_link();
    } else if( op.prob_case(NEAT::mutate_delete_node_prob) ) {
        mutate_delete_node();
    } else {
        //Only do other mutations when not doing sturctural mutations
        if( rng.under(NEAT::mutate_random_trait_prob) ) {
            mutate_random_trait();
        }
        if( rng.under(NEAT::mutate_link_trait_prob) ) {
            mutate_link_trait(1);
        }
        if( rng.under(NEAT::mutate_node_trait_prob) ) {
            mutate_node_trait(1);
        }
        if( rng.under(NEAT::mutate_link_weights_prob) ) {
            mutate_link_weights(NEAT::weight_mut_power,1.0,GAUSSIAN);
        }
        if( rng.under(NEAT::mutate_toggle_enable_prob) ) {
            mutate_toggle_enable(1);
        }
        if (rng.under(NEAT::mutate_gene_reenable_prob) ) {
            mutate_gene_reenable(); 
        }
    }
}

void InnovGenome::mutate_toggle_enable(int times) {
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
	//Search for a disabled gene
    for(InnovLinkGene &g: links) {
        if(!g.enable) {
            g.enable = true;
            break;
        }
    }
}

void InnovGenome::mate(CreateInnovationFunc create_innov,
                       InnovGenome *genome1,
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

    //Determine whether to mutate the baby's InnovGenome
    //This is done randomly or if the genome1 and genome2 are the same organism
    if( !offspring->rng.under(NEAT::mate_only_prob) ||
        (genome2->genome_id == genome1->genome_id) ||
        (genome2->compatibility(genome1) == 0.0) ) {

        offspring->mutate(create_innov);
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

*/
