#include "std.h" // Must be included first. Precompiled header with standard library includes.
#include "networkmanager.h"
#include "cpunetworkmanager.h"

using namespace NEAT;

NetworkManager *NetworkManager::create() {
    return new CpuNetworkManager();
}
