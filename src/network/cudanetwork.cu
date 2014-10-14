#define DEVICE_CODE

#include <iostream>
#include <vector>

#include "cudanetwork.h"

#include <assert.h>
#include <cuda.h>
#include <limits.h>
#include <stdio.h>

//--------------------------------------------------------------------------------
//---
//--- MACROS
//---
//--------------------------------------------------------------------------------
#define errif( STMT, MSG... ) if( STMT ) { fprintf(stderr, "[%s:%d] '%s' ", __FILE__, __LINE__, #STMT); fprintf(stderr, MSG); fprintf(stderr, "\n"); abort(); }
#define require( STMT ) if( !(STMT) ) { fprintf(stderr, "ASSERTION ERROR! [%s:%d] '%s'\n", __FILE__, __LINE__, #STMT); abort(); }
#define panic() { fprintf(stderr, "PANIC! [%s:%d]\n", __FILE__, __LINE__); abort(); }
#define trap(msg) {std::cerr << __FILE__ << ":" << __LINE__ << ": " << msg << std::endl; abort();}

#define p(msg) std::cout << "[cuda]: " << msg << std::endl

#define Threads_Per_Block 64
#define MAX_NEURONS Threads_Per_Block
#define NACTIVATE_ITERATIONS 10

#define xcuda(stmt) {                                                   \
        cudaError_t err = stmt;                                         \
        if (err != cudaSuccess) {                                       \
            std::cerr << __FILE__ << ":" << __LINE__ << ": Failed to run " << #stmt << ". Reason: " << cudaGetErrorString(err) << std::endl; \
            exit(1);                                                    \
        }                                                               \
    }

namespace NEAT {
    __global__ void activate(GpuState *states,
                             RawBuffers bufs,
                             uint ncycles);


    static uchar *alloc_host(uint size) {
        uchar *result;
        xcuda( cudaMallocHost((void **)&result, size) );
        return result;
    }
    static uchar *alloc_dev(uint size) {
        uchar *result;
        xcuda( cudaMalloc((void **)&result, size) );
        return result;
    }
    static void free_host(__inout uchar *&buf) {
        if(buf) {
            xcuda( cudaFreeHost(buf) );
            buf = 0;
        }
    }
    static void free_dev(__inout uchar *&buf) {
        if(buf) {
            xcuda( cudaFree(buf) );
            buf = 0;
        }
    }
    static void grow_buffers(__inout uchar *&h_buf, __inout uchar *&d_buf,
                             __inout uint &capacity, __in uint newlen) {
        free_host(h_buf);
        free_dev(d_buf);
        capacity = newlen;
        h_buf = alloc_host(newlen);
        d_buf = alloc_dev(newlen);
    }

//--------------------------------------------------------------------------------
//---
//--- CLASS CudaNetworkBatch
//---
//--------------------------------------------------------------------------------
    CudaNetworkBatch::CudaNetworkBatch(uint nnets_)
        : nnets(nnets_) {
        memset(&h_bufs, 0, sizeof(h_bufs));
        memset(&d_bufs, 0, sizeof(d_bufs));
        memset(&offsets, 0, sizeof(offsets));
        memset(&capacity, 0, sizeof(capacity));
        memset(&lens, 0, sizeof(lens));

        h_gpu_states = (GpuState *)alloc_host(sizeof(GpuState) * nnets);
        d_gpu_states = (GpuState *)alloc_dev(sizeof(GpuState) * nnets);
    }

    CudaNetworkBatch::~CudaNetworkBatch() {
        free_host((uchar *&)h_gpu_states);
        free_dev((uchar *&)d_gpu_states);

        free_host(h_bufs.main);
        free_host(h_bufs.input);
        free_host(h_bufs.output);

        free_dev(d_bufs.main);
        free_dev(d_bufs.input);
        free_dev(d_bufs.output);
    }

    void CudaNetworkBatch::configure(CudaNetwork **nets,
                                     uint nnets) {
        assert(nnets = this->nnets);

        memset(&lens, 0, sizeof(lens));
        sizeof_shared = 0;

        Offsets nets_offs[nnets];

        for(uint i = 0; i < nnets; i++) {
            CudaNetwork &net = *nets[i];
            CudaNetDims &dims = net.dims;

            Lens net_lens;
            Offsets &net_offs = nets_offs[i];
            uint net_sizeof_shared =
                (2 * sizeof(real_t) * dims.nnodes.all)
                + (sizeof(real_t) * Threads_Per_Block);

            //main buffer
            {
                uint sizeof_activation = sizeof(real_t) * dims.nnodes.hidden;
                uint sizeof_links = sizeof(CudaLink) * dims.nlinks;
                uint sizeof_partitions = sizeof(ActivationPartition) * dims.npartitions;

                net_lens.main = sizeof_activation + sizeof_links + sizeof_partitions;

                net_offs.main.activation = this->lens.main;
                net_offs.main.links = net_offs.main.activation + sizeof_activation;
                net_offs.main.partitions = net_offs.main.links + sizeof_links;
            }

            //input buffer
            {
                uint sizeof_parms = sizeof(ActivateParms);
                uint sizeof_activation = sizeof(real_t) * dims.nnodes.sensor;

                net_lens.input = sizeof_parms + sizeof_activation;

                net_offs.input.activate_parms = this->lens.input;
                net_offs.input.activation = net_offs.input.activate_parms + sizeof_parms;
            }

            //output buffer
            {
                uint sizeof_activation = sizeof(real_t) * dims.nnodes.output;

                net_lens.output = sizeof_activation;

                net_offs.output.activation = this->lens.output;
            }

            sizeof_shared = max(sizeof_shared, net_sizeof_shared);
            
            net.offsets = net_offs;
            lens += net_lens;
        }

        if(lens.main > capacity.main) {
            uint newlen = uint(lens.main * 1.4);
            p("alloc main: " << newlen);
            grow_buffers(h_bufs.main, d_bufs.main, capacity.main, newlen);
        }
        if(lens.input > capacity.input) {
            uint newlen = uint(lens.input);
            p("alloc input: " << newlen);
            assert(capacity.input == 0); // should only alloc once
            grow_buffers(h_bufs.input, d_bufs.input, capacity.input, newlen);
        }
        if(lens.output > capacity.output) {
            uint newlen = uint(lens.output);
            p("alloc output: " << newlen);
            assert(capacity.output == 0); // should only alloc once
            grow_buffers(h_bufs.output, d_bufs.output, capacity.output, newlen);
        }

        for(uint i = 0; i < nnets; i++) {
            CudaNetwork *net = nets[i];
            net->set_bufs(h_bufs, nets_offs[i]);

            GpuState &gpu = h_gpu_states[i];
            gpu.dims = net->dims;
            gpu.offsets = net->offsets;
        }

        xcuda( cudaMemcpy(d_gpu_states, h_gpu_states, sizeof(GpuState) * nnets, cudaMemcpyHostToDevice) );
        xcuda( cudaMemcpy(d_bufs.main, h_bufs.main, lens.main, cudaMemcpyHostToDevice) );
    }

    void CudaNetworkBatch::activate(uint ncycles) {
        xcuda( cudaMemcpy(d_bufs.input,
                          h_bufs.input,
                          lens.input,
                          cudaMemcpyHostToDevice) );

        NEAT::activate<<<nnets, Threads_Per_Block, sizeof_shared>>>(d_gpu_states,
                                                                    d_bufs,
                                                                    ncycles);
        
        xcuda( cudaMemcpy(h_bufs.output,
                          d_bufs.output,
                          lens.output,
                          cudaMemcpyDeviceToHost) );
    }

//--------------------------------------------------------------------------------
//---
//--- CLASS CudaNetwork
//---
//--------------------------------------------------------------------------------
#define __dh_util static inline __device__ __host__

    __dh_util CudaLink *links(const RawBuffers &bufs,
                                  const Offsets &offs) {
        return (CudaLink *)(bufs.main + offs.main.links);
    }

    __dh_util ActivationPartition *partitions(const RawBuffers &bufs,
                                                  const Offsets &offs) {
        return (ActivationPartition *)(bufs.main + offs.main.partitions);
    }

    __dh_util real_t *input_activations(const RawBuffers &bufs,
                                        const Offsets &offs) {
        return (real_t *)(bufs.input + offs.input.activation);
    }

    __dh_util real_t *hidden_activations(const RawBuffers &bufs,
                                             const Offsets &offs) {
        return (real_t *)(bufs.main + offs.main.activation);
    }

    __dh_util real_t *output_activations(const RawBuffers &bufs,
                                             const Offsets &offs) {
        return (real_t *)(bufs.output + offs.output.activation);
    }

    __dh_util ActivateParms &activate_parms(const RawBuffers &bufs,
                                                const Offsets &offs) {
        return *(ActivateParms *)(bufs.input + offs.input.activate_parms);
    }

#undef __dh_util

    void CudaNetwork::set_bufs(const RawBuffers &bufs_,
                               const Offsets &offsets_) {
        bufs = bufs_;
        offsets = offsets_;

        memcpy( NEAT::links(bufs, offsets),
                gpu_links.data(),
                sizeof(CudaLink) * gpu_links.size() );
        memcpy( NEAT::partitions(bufs, offsets),
                partitions.data(),
                sizeof(ActivationPartition) * partitions.size() );

        activate_parms(bufs, offsets).clear_noninput = true;
        activate_parms(bufs, offsets).enabled = true;
    }

    void CudaNetwork::disable() {
        activate_parms(bufs, offsets).enabled = false;
    }

    bool CudaNetwork::is_enabled() {
        return activate_parms(bufs, offsets).enabled;
    }

    void CudaNetwork::configure(const NetDims &dims_,
                                NetNode *nodes,
                                NetLink *links) {

        static_cast<NetDims &>(dims) = dims_;

        partitions.clear();
        gpu_links.resize(dims.nlinks);

        if(dims.nlinks != 0) {
            ActivationPartition partition;

            for(link_size_t i = 0; i < dims.nlinks; i++) {
                NetLink &link = links[i];
                if( (i % Threads_Per_Block == 0)
                    || (link.out_node_index != partition.out_node_index) ) {

                    if(i != 0) {
                        partitions.push_back(partition);
                    }

                    partition.out_node_index = link.out_node_index;
                    partition.offset = i % Threads_Per_Block;
                    partition.len = 0;
                }
                partition.len++;

                CudaLink &gpu_link = gpu_links[i];
                gpu_link.in_node_index = link.in_node_index;
                gpu_link.partition = partitions.size();
                gpu_link.weight = link.weight;
            }

            partitions.push_back(partition);
        }
        dims.npartitions = partitions.size();
    }

    void CudaNetwork::load_sensors(const std::vector<real_t> &sensvals,
                                   bool clear_noninput) {
        memcpy( input_activations(bufs, offsets),
                sensvals.data(),
                sizeof(real_t) * dims.nnodes.sensor );

        if(clear_noninput) {
            activate_parms(bufs, offsets).clear_noninput = clear_noninput;
        }
    }

    real_t CudaNetwork::get_output(size_t index) {
        return output_activations(bufs, offsets)[index];
    }

//--------------------------------------------------------------------------------
//---
//--- GPU KERNEL CODE
//---
//--------------------------------------------------------------------------------
    __device__ void sum_partition(float *x, int i, int n, float *result) {
        int stride = __popc(n) == 1 ? n >> 1 : 1 << 31 - __clz(n);

        if(i + stride < n) {
            x[i] += x[i + stride];
        }
      
        __syncthreads();

        stride >>= 1;
        // max_stride necessary to keep all threads from all partitions in sync.
        for(int max_stride = Threads_Per_Block >> 4; max_stride > 0; stride >>= 1, max_stride >>= 1) {
            if(i < stride) {
                x[i] += x[i + stride];
            }
            __syncthreads();
        }

        if(i == 0) {
            *result += x[0];
        }

        __syncthreads();
    }

    static __device__ float logistic(float x) {
        return (1.0f / (1.0f + exp(-1 * x * 4.924273)));
    }

    __global__ void activate(GpuState *states,
                             RawBuffers bufs,
                             uint ncycles) {
        GpuState state = states[blockIdx.x];
        if(!activate_parms(bufs, state.offsets).enabled) {
            return;
        }

        extern __shared__ char __shared_buf[];

        real_t *activation = (real_t *)__shared_buf;
        real_t *newactivation = activation + state.dims.nnodes.all;
        real_t *partial_activation = newactivation + state.dims.nnodes.all;

        uint tid = threadIdx.x;

        for(uint inode = tid; inode < state.dims.nnodes.all; inode += Threads_Per_Block) {
            if(inode < state.dims.nnodes.input) {
                const uint nbias = state.dims.nnodes.bias;
                if(inode < nbias) {
                    activation[inode] = 1.0;
                } else {
                    activation[inode] =
                        input_activations(bufs, state.offsets)[inode - nbias];
                }
                newactivation[inode] = activation[inode];
            } else {
                const uint nio = state.dims.nnodes.input + state.dims.nnodes.output;
                if( activate_parms(bufs, state.offsets).clear_noninput ) {
                    activation[inode] = 0.0;
                } else {
                    activation[inode] =
                        hidden_activations(bufs, state.offsets)[inode - nio];
                }
            }
        }
        __syncthreads();

        const int nits = 1 + (state.dims.nlinks - 1) / Threads_Per_Block;

        for(uint icycle = 0; icycle < ncycles; icycle++) {
            for(uint ilink = tid, it = 0; it < nits; ilink += Threads_Per_Block, it++) {
                float *partition_x;
                int partition_i;
                int partition_n;
                float *result;

                if(ilink < state.dims.nlinks) {
                    CudaLink link = links(bufs, state.offsets)[ilink];
                    partial_activation[tid] = link.weight * activation[link.in_node_index];

                    ActivationPartition p = partitions(bufs, state.offsets)[link.partition];
                    partition_x = partial_activation + p.offset;
                    partition_i = tid - p.offset;
                    partition_n = p.len;
                    result = newactivation + p.out_node_index;
                } else {
                    partition_x = NULL;
                    partition_i = 1;
                    partition_n = 0;
                    result = NULL;
                }
                __syncthreads();

                sum_partition(partition_x,
                              partition_i,
                              partition_n,
                              result);

                __syncthreads();
            }

            for(uint inode = tid; inode < state.dims.nnodes.all; inode += Threads_Per_Block) {
                if(inode >= state.dims.nnodes.input) {
                    newactivation[inode] = logistic( newactivation[inode] );
                }
            }
            {
                float *swap = newactivation;
                newactivation = activation;
                activation = swap;
            }
            __syncthreads();
        }

        const uint nio = state.dims.nnodes.input + state.dims.nnodes.output;

        for(uint inode = tid + state.dims.nnodes.input;
            inode < state.dims.nnodes.all;
            inode += Threads_Per_Block) {

            if(inode < nio) {
                output_activations(bufs, state.offsets)[inode - state.dims.nnodes.input] = activation[inode];
            } else {
                hidden_activations(bufs, state.offsets)[inode - nio] = activation[inode];
            }
        }
    }

} // namespace NEAT


//--------------------------------------------------------------------------------
//---
//--- OLD STUFF
//---
//--------------------------------------------------------------------------------
#if false
__device__ void sum_partition(float *x, int i, int n, float *result) {
    int stride = __popc(n) == 1 ? n >> 1 : 1 << 31 - __clz(n);

    if(i + stride < n) {
        x[i] += x[i + stride];
    }
      
    __syncthreads();

    stride >>= 1;
    // max_stride necessary to keep all threads from all partitions in sync.
    for(int max_stride = Threads_Per_Block >> 4; max_stride > 0; stride >>= 1, max_stride >>= 1) {
        if(i < stride) {
            x[i] += x[i + stride];
        }
        __syncthreads();
    }

    if(i == 0) {
        *result += x[0];
    }

    __syncthreads();
}

static __device__ float logistic(float x) {
    return (1.0f / (1.0f + exp(-1 * x * 4.924273)));
}

__global__ void update(FiringRateModel_Cuda::GpuState *states) {
    int tid = threadIdx.x;

    FiringRateModel_Cuda::GpuState state = states[blockIdx.x];

    extern __shared__ char __shared_buf[];

    float *neuronactivation = (float *)__shared_buf;
    float *newneuronactivation = neuronactivation + state.neurons_count;
    float *partial_activation = newneuronactivation + state.neurons_count;

    if(tid < state.neurons_count) {
    	if(tid < state.input_neurons_count) {
        	neuronactivation[tid] = state.buffers.input_activation[tid];
            newneuronactivation[tid] = neuronactivation[tid];
    	} else {
			neuronactivation[tid] = state.activations()[tid];
		}
    }
    __syncthreads();

    for(int it = 0; it < NACTIVATE_ITERATIONS; it++) {

        for(int isynapse = tid; isynapse < state.synapses_count; isynapse += Threads_Per_Block) {
            float *partition_x;
            int partition_i;
            int partition_n;
            float *result;

            if(isynapse < state.synapses_count) {
                CudaSynapse synapse = state.synapses()[isynapse];
                partial_activation[tid] = synapse.efficacy * neuronactivation[synapse.fromneuron];

                NeuronActivationPartition p = state.partitions()[synapse.partition];
                partition_x = partial_activation + p.offset;
                partition_i = tid - p.offset;
                partition_n = p.len;
                result = newneuronactivation + p.toneuron;
            } else {
                partition_x = NULL;
                partition_i = 1;
                partition_n = 0;
                result = NULL;
            }
            __syncthreads();

            sum_partition(partition_x,
                          partition_i,
                          partition_n,
                          result);

            __syncthreads();
        }

        if( (tid >= state.input_neurons_count) && (tid < state.neurons_count) ) {
            newneuronactivation[tid] = logistic( newneuronactivation[tid] );
        }
        {
            float *swap = newneuronactivation;
            newneuronactivation = neuronactivation;
            newneuronactivation = swap;
        }
        __syncthreads();
    }

    if(tid < state.neurons_count) {
        state.activations()[tid] = neuronactivation[tid];

        if( (tid >= state.input_neurons_count)
            && (tid < state.input_neurons_count + state.output_neurons_count) ) {
            state.buffers.output_activation[tid - state.input_neurons_count] = neuronactivation[tid];
        }
    }
}

typedef FiringRateModel_Cuda::AgentState AgentState;
typedef FiringRateModel_Cuda::GpuState GpuState;

static GpuState *gpus = NULL;
static GpuState *d_gpus = NULL;
static unsigned char *buffers = NULL;
static unsigned char *d_buffers = NULL;
static uint sizeof_shared = 0;
static float *d_all_input = NULL;
static uint sizeof_input = 0;
static float *d_all_output = NULL;
static uint sizeof_output = 0;

void FiringRateModel_Cuda::update_all(AgentState *agents,
                                      long nagents,
                                      float *all_input,
                                      float *all_output) {

    xcuda( cudaMemcpy(d_all_input,
                      all_input,
                      sizeof_input,
                      cudaMemcpyHostToDevice) );

    ::update<<<nagents, Threads_Per_Block, sizeof_shared>>>(d_gpus);

    xcuda( cudaMemcpy(all_output,
                      d_all_output,
                      sizeof_output,
                      cudaMemcpyDeviceToHost) );
}
#endif // #if false (old stuff)
