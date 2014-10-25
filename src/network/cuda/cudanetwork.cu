#include "cudanetwork.h"

namespace NEAT {
 
    Network *Network::create() {
        return new CudaNetwork();
    }

}
