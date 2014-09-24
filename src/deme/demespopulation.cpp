#include "demespopulation.h"

#include "genomemanager.h"
#include "timer.h"
#include "util.h"
#include <algorithm>

using namespace NEAT;
using namespace std;

//todo: put in env
#define NUM_GLOBAL_ELITES 5
#define SEND_GENERATION_ELITES true

static bool cmp_org_desc(const Organism *a, const Organism *b) {
    return a->fitness > b->fitness;
}

static void sort_elites(vector<Organism *> &elites) {
    std::sort(elites.begin(), elites.end(), cmp_org_desc);
}

static int insert_if_elite(vector<Organism *> &elites, Organism *candidate) {
    auto it = std::lower_bound(elites.begin(), elites.end(), candidate, cmp_org_desc);
    if(it == elites.end())
        return -1;

    size_t i = it - elites.begin();
    Organism *free_slot = elites.back();

    assert(candidate->fitness >= free_slot->fitness);
    assert(candidate->fitness >= elites[i]->fitness);
    assert(i == 0 || candidate->fitness <= elites[i-1]->fitness);

    *free_slot = *candidate;

    elites.erase(elites.end() - 1);
    elites.insert(elites.begin() + i, free_slot);

    assert(elites[i] == free_slot);

    return i;
}

DemesPopulation::DemesPopulation(rng_t rng,
                                 GenomeManager *genome_manager_,
                                 vector<unique_ptr<Genome>> &seeds)
    : generation(0)
    , genome_manager(genome_manager_) {

    size_t pop_size = seeds.size();
    assert(pop_size % NEAT::deme_count == 0);

    size_t deme_size = pop_size / NEAT::deme_count;

    demes.reserve(NEAT::deme_count);
    for(size_t i = 0; i < (size_t)NEAT::deme_count; i++) {
        rng_t deme_rng;
        deme_rng.seed(rng.integer());
        demes.emplace_back(deme_rng,
                           seeds,
                           deme_size,
                           i * deme_size);
    }

    for(size_t i = 0; i < NUM_GLOBAL_ELITES; i++) {
        global_elites.push_back(new Organism(*seeds.front()));
    }
}

DemesPopulation::~DemesPopulation() {
}

Organism &DemesPopulation::get_fittest() {
    return *global_elites.front();
}

bool DemesPopulation::evaluate(std::function<void (Organism &org)> eval_org) {
    static Timer timer("evaluate");
    timer.start();

    real_t old_fitness = global_elites.front()->fitness;

    for(Deme &deme: demes) {
        deme.evaluate(eval_org);
    }

    elites.clear();
    for(Deme &deme: demes) {
        elites.push_back( &deme.get_fittest() );
    }
    sort_elites(elites);

    bool new_fittest = false;
    for(Organism *elite: elites) {
        int index = insert_if_elite(global_elites, elite);
        if(index < 0) {
            break;
        } else if(index == 0) {
            new_fittest = true;
        }
    }

#if !SEND_GENERATION_ELITES
    elites.clear();
#endif
    for(Organism *elite: global_elites) {
        elites.push_back(elite);
    }

    timer.stop();

    if(new_fittest) {
        real_t new_fitness = global_elites.front()->fitness;
        real_t delta = new_fitness - old_fitness;
        if(delta > 0) {
            cout << "new fittest: " << new_fitness << ", delta: " << (new_fitness - old_fitness) << " @ gen " << generation << endl;
        } else {
            // In the case of a fitness tie, we let the new guy in, for the sake of
            // diversity, but we don't really want to report it.
            new_fittest = false;
        }
    }

    return new_fittest;
}

void DemesPopulation::next_generation() {
    static Timer timer("reproduce");
    timer.start();

    generation++;

    for(Deme &deme: demes) {
        deme.next_generation(elites, genome_manager);
    }

    genome_manager->finalize_generation();

    for(Deme &deme: demes) {
        deme.create_phenotypes();
    }

    timer.stop();
}

void DemesPopulation::write(std::ostream& out) {
    for(Deme &deme: demes) {
        deme.write(out);
    }
}

void DemesPopulation::verify() {
    for(Deme &deme: demes) {
        deme.verify();
    }
} 



