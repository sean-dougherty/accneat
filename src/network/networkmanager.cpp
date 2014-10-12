#include "std.h" // Must be included first. Precompiled header with standard library includes.
#include "networkmanager.h"
#include "cpunetwork.h"
#include "network.h"
#include "util.h"

using namespace NEAT;
using namespace std;

unique_ptr<Network> NetworkManager::make_default() {
    return unique_ptr<Network>(new CpuNetwork());
}

