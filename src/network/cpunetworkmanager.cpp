#include "std.h" // Must be included first. Precompiled header with standard library includes.
#include "networkmanager.h"
#include "cpunetwork.h"
#include "util.h"

using namespace NEAT;
using namespace std;

#define NACTIVATES_PER_INPUT 10

unique_ptr<Network> CpuNetworkManager::make_default() {
    return unique_ptr<Network>(new CpuNetwork());
}
