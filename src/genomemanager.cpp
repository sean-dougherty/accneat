#include "genomemanager.h"

#include "innovgenomemanager.h"

using namespace NEAT;

GenomeManager *GenomeManager::create() {
    return new InnovGenomeManager();
}
