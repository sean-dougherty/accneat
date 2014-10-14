#include "std.h" // Must be included first. Precompiled header with standard library includes.
#include "networkmanager.h"

#ifdef ENABLE_CUDA
#include "cudanetworkmanager.h"
#else
#include "cpunetworkmanager.h"
#endif

using namespace NEAT;

NetworkManager *NetworkManager::create() {
#ifdef ENABLE_CUDA
    return new CudaNetworkManager();
#else
    return new CpuNetworkManager();
#endif
}
