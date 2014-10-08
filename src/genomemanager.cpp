#include "std.h" // Must be included first. Precompiled header with standard library includes.
#include "genomemanager.h"
#include "innovgenomemanager.h"
#include "util.h"

using namespace NEAT;

GenomeManager *GenomeManager::create() {
    switch(env->genome_type) {
    case GenomeType::INNOV:
        return new InnovGenomeManager();
    default:
        panic();
    }
}
