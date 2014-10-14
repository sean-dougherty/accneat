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

#define Threads_Per_Block 512
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

//--------------------------------------------------------------------------------
//---
//--- CLASS CudaNetworkBatch
//---
//--------------------------------------------------------------------------------
    CudaNetworkBatch::CudaNetworkBatch(uint nnets_)
        : nnets(nnets_) {

        memset(&template_dims, 0, sizeof(template_dims));
        memset(&h_bufs, 0, sizeof(h_bufs));
        memset(&d_bufs, 0, sizeof(d_bufs));
        memset(&offsets, 0, sizeof(offsets));
        memset(&capacity, 0, sizeof(capacity));
        memset(&lens, 0, sizeof(lens));

        h_gpu_states = (GpuState *)alloc_host(sizeof(GpuState) * nnets);
        d_gpu_states = (GpuState *)alloc_dev(sizeof(GpuState) * nnets);
    }

    CudaNetworkBatch::~CudaNetworkBatch() {
        panic();

        free_host((uchar *&)h_gpu_states);
        free_dev((uchar *&)d_gpu_states);

        free_host(h_bufs.main);
        free_host(h_bufs.input);
        free_host(h_bufs.output);

        free_dev(d_bufs.main);
        free_dev(d_bufs.input);
        free_dev(d_bufs.output);
    }

    void CudaNetworkBatch::configure_net(__in CudaNetDims &dims,
                                         __out RawBuffers &bufs,
                                         __out Offsets &offsets) {
        bool init;

        if(!template_dims.nnodes.all) {
            init = true;
            template_dims = dims;
        } else {
            init = false;
            assert(dims.nnodes.bias == template_dims.nnodes.bias);
            assert(dims.nnodes.sensor == template_dims.nnodes.sensor);
            assert(dims.nnodes.output == template_dims.nnodes.output);
        }

        Lens required;
        required.main =
            (sizeof(real_t) * dims.nnodes.hidden)
            + (sizeof(CudaLink) * dims.nlinks)
            + (sizeof(ActivationPartition) * dims.npartitions);

        required.input =
            sizeof(ActivateParms)
            + (sizeof(real_t) * dims.nnodes.sensor);

        required.output =
            (sizeof(real_t) * dims.nnodes.output);

        Lens new_lens = lens + required;

        if(init) {
            Lens new_capacity;
            new_capacity.input = required.input * nnets;
            new_capacity.output = required.output * nnets;
            new_capacity.main = uint(required.main * nnets * 1.5);

            h_bufs.input = alloc_host(new_capacity.input);
            h_bufs.output = alloc_host(new_capacity.output);
            h_bufs.main = alloc_host(new_capacity.main);

            capacity = new_capacity;
        } else if(new_lens > capacity) {
            free_dev(__inout d_bufs.main);

            Lens new_capacity = capacity;
            new_capacity.main = uint(capacity.main * 1.5);

            uchar *new_main = alloc_host(new_capacity.main);
            memcpy(new_main, h_bufs.main, lens.main);
            free_host(h_bufs.main);
            h_bufs.main = new_main;

            capacity = new_capacity;
        }

        bufs = h_bufs;

        offsets.main.activation = lens.main;
        offsets.main.links = offsets.main.activation + (sizeof(real_t) * dims.nnodes.hidden);
        offsets.main.partitions = offsets.main.links + (sizeof(CudaLink) * dims.nlinks);
        offsets.input.activate_parms = lens.input;
        offsets.input.activation = offsets.input.activate_parms + sizeof(ActivateParms);
        offsets.output.activation = lens.output;
        
        lens = new_lens;
    }

    void CudaNetworkBatch::configure_device(CudaNetwork **nets,
                                            uint nnets) {
        assert(nnets = this->nnets);

        if(!d_bufs.input) {
            d_bufs.input = alloc_dev(capacity.input);
            assert(!d_bufs.output);
            d_bufs.output = alloc_dev(capacity.output);
        }
        if(!d_bufs.main) {
            d_bufs.main = alloc_dev(capacity.main);
        }

        for(uint i = 0; i < nnets; nnets++) {
            CudaNetwork *net = nets[i];
            GpuState &gpu = h_gpu_states[i];

            net->get_gpu(gpu);
            net->set_bufs(h_bufs);
        }
        xcuda( cudaMemcpy(d_gpu_states, h_gpu_states, sizeof(GpuState) * nnets, cudaMemcpyHostToDevice) );

        xcuda( cudaMemcpy(d_bufs.main, h_bufs.main, lens.main, cudaMemcpyHostToDevice) );

        step = 1;
    }

    void CudaNetworkBatch::activate() {
        xcuda( cudaMemcpy(d_bufs.input,
                          h_bufs.input,
                          lens.input,
                          cudaMemcpyHostToDevice) );

        //::update<<<nagents, Threads_Per_Block, sizeof_shared>>>(d_gpus);
        
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
    static inline CudaLink *links(const RawBuffers &bufs,
                                  const Offsets &offs) {
        return (CudaLink *)(bufs.main + offs.main.links);
    }

    static inline ActivationPartition *partitions(const RawBuffers &bufs,
                                                  const Offsets &offs) {
        return (ActivationPartition *)(bufs.main + offs.main.partitions);
    }

    static inline real_t *input_activations(const RawBuffers &bufs,
                                            const Offsets &offs) {
        return (real_t *)(bufs.input + offs.input.activation);
    }

    static inline real_t *output_activations(const RawBuffers &bufs,
                                             const Offsets &offs) {
        return (real_t *)(bufs.output + offs.output.activation);
    }

    static inline ActivateParms &activate_parms(const RawBuffers &bufs,
                                                const Offsets &offs) {
        return *(ActivateParms *)(bufs.input + offs.input.activate_parms);
    }

    void CudaNetwork::get_gpu(__out GpuState &gpu) {
        gpu.dims = dims;
        gpu.offsets = offsets;
    }

    void CudaNetwork::set_bufs(const RawBuffers &bufs_) {
        bufs = bufs_;
        activate_parms(bufs, offsets).flush_step = 1;
    }

    void CudaNetwork::configure(const NetDims &dims_,
                                NetNode *nodes,
                                NetLink *links) {

        static_cast<NetDims &>(dims) = dims_;
        dims.npartitions = 0;

        ActivationPartition partitions[dims.nlinks];
        CudaLink gpu_links[dims.nlinks];

        if(dims.nlinks == 0) {
            dims.npartitions = 0;
        } else {
            ActivationPartition *currpartition = NULL;

            for(link_size_t i = 0; i < dims.nlinks; i++) {
                NetLink &link = links[i];
                if( (i % Threads_Per_Block == 0)
                    || (link.out_node_index != currpartition->out_node_index) ) {
                    if(currpartition)
                        currpartition++;
                    else
                        currpartition = partitions;
                    assert(currpartition - partitions < dims.nlinks);

                    currpartition->out_node_index = link.out_node_index;
                    currpartition->offset = i % Threads_Per_Block;
                    currpartition->len = 0;
                }
                currpartition->len++;

                CudaLink &gpu_link = gpu_links[i];
                gpu_link.in_node_index = link.in_node_index;
                gpu_link.partition = currpartition - partitions;
                gpu_link.weight = link.weight;
            }

            dims.npartitions = currpartition - partitions + 1;
        }

        RawBuffers bufs;
        batch->configure_net(this->dims,
                             __out bufs,
                             __out this->offsets);

        memcpy( NEAT::links(bufs, offsets),
                gpu_links,
                sizeof(CudaLink) * dims.nlinks );
        
        memcpy( NEAT::partitions(bufs, offsets),
                partitions,
                sizeof(ActivationPartition) * dims.npartitions );
    }

    void CudaNetwork::load_sensors(const std::vector<real_t> &sensvals,
                                   bool clear_noninput) {
        memcpy( input_activations(bufs, offsets),
                sensvals.data(),
                sizeof(real_t) * dims.nnodes.sensor );

        if(clear_noninput) {
            activate_parms(bufs, offsets).flush_step = batch->get_step();
        }
    }

    real_t CudaNetwork::get_output(size_t index) {
        return output_activations(bufs, offsets)[index];
    }

#if false

    void CudaNetwork::init_activations() {
        for(size_t j = 0; j < gpu.dims.nnodes.bias; j++) {
            host.input_activation[j] = 1.0;
        }
        host.update_parms->flush = true;
    }

    CudaNetworkBatch::CudaNetworkBatch(CudaNetwork **nets, size_t nnets) {
        assert(nnets > 0);

        node_size_t ninput_nodes = nets[0]->gpu.dims.nnodes.input;
        node_size_t noutput_nodes = nets[0]->gpu.dims.nnodes.output;            

        // update_input
        {
            uint sizeof_input_activations = nnets * ninput_nodes * sizeof(real_t);
            uint sizeof_parms = nnets * sizeof(CudaNetwork::UpdateParms);

            update_input.buflen = sizeof_input_activations + sizeof_parms;
            update_input.offset_activations = 0;
            update_input.offset_parms = sizeof_input_activations;

            xcuda( cudaMallocHost((void **)&update_input.buffer, update_input.buflen) );
            xcuda( cudaMalloc((void **)&update_input.d_buffer, update_input.buflen) );
        }

        // update_output
        {
            uint sizeof_output_activations = nnets * noutput_nodes * sizeof(real_t);
            
            update_output.buflen = sizeof_output_activations;
            
            xcuda( cudaMallocHost((void **)&update_output.buffer, update_output.buflen) );
            xcuda( cudaMalloc((void **)&update_output.d_buffer, update_output.buflen) );
        }

        // config
        {
            uint sizeof_gpus = nnets * sizeof(CudaNetwork::GpuState);
            xcuda( cudaMallocHost((void **)&config.gpus, sizeof_gpus) );
            xcuda( cudaMalloc((void **)&config.d_gpus, sizeof_gpus) );

            uint sizeof_buffers = 0;
            sizeof_shared = 0;
            for(size_t i = 0; i < nnets; i++) {
                CudaNetwork *net = nets[i];

                sizeof_buffers += net->buffer.size();

                uint sizeof_agent_shared =
                    (2 * net->gpu.dims.nnodes.all * sizeof(real_t)) // activation
                    + (Threads_Per_Block * sizeof(real_t)); // partial_activation

                sizeof_shared = max(sizeof_shared, sizeof_agent_shared);
            }
            xcuda( cudaMallocHost((void **)&config.buffers, sizeof_buffers) );
            xcuda( cudaMalloc((void **)&config.d_buffers, sizeof_buffers) );

            real_t *host_input_activations =
                (real_t *)(update_input.buffer + update_input.offset_activations);
            CudaNetwork::UpdateParms *host_parms =
                (CudaNetwork::UpdateParms *)(update_input.buffer + update_input.offset_parms);
            real_t *host_output_activations =
                (real_t *)(update_output.buffer);

            real_t *d_input_activations =
                (real_t *)(update_input.d_buffer + update_input.offset_activations);
            CudaNetwork::UpdateParms *d_parms =
                (CudaNetwork::UpdateParms *)(update_input.d_buffer + update_input.offset_parms);
            real_t *d_output_activations =
                (real_t *)(update_output.d_buffer);

            uint buffers_offset = 0;
            for(size_t i = 0; i < nnets; i++) {
                CudaNetwork *net = nets[i];

                net->host.input_activation = host_input_activations + (i * ninput_nodes);
                net->host.update_parms = host_parms + i;
                net->host.output_activation = host_output_activations + (i * noutput_nodes);

                net->init_activations();

                net->gpu.buffers.__main = config.d_buffers + buffers_offset;
                net->gpu.buffers.input_activation = d_input_activations + (i * ninput_nodes);
                net->gpu.buffers.update_parms = d_parms + i;
                net->gpu.buffers.output_activation = d_output_activations + (i * ninput_nodes);

                config.gpus[i] = net->gpu;
                memcpy(config.buffers, net->buffer.data(), net->buffer.size());

                buffers_offset += net->buffer.size();
            }

            xcuda( cudaMemcpy(config.d_gpus, config.gpus, sizeof_gpus, cudaMemcpyHostToDevice) );
            xcuda( cudaMemcpy(config.d_buffers, config.buffers, sizeof_buffers, cudaMemcpyHostToDevice) );
        }
    }
#endif

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
