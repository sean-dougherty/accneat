#define DEVICE_CODE

#include <iostream>
#include <vector>

#include "cudanet.h"

#include <assert.h>
#include <cuda.h>
#include <limits.h>
#include <stdio.h>

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

FiringRateModel_Cuda::FiringRateModel_Cuda() {
}

FiringRateModel_Cuda::~FiringRateModel_Cuda() {
    delete [] buffer;
}

void FiringRateModel_Cuda::init(FiringRateModel__Neuron *neurons,
                                short neurons_count,
                                short input_neurons_count,
                                short output_neurons_count,
                                float *neuronactivation,
                                FiringRateModel__Synapse *synapses,
                                long synapses_count) {

    require(neurons_count < MAX_NEURONS);

    gpu.neurons_count = neurons_count;
    gpu.input_neurons_count = input_neurons_count;
    gpu.output_neurons_count = output_neurons_count;
    gpu.synapses_count = synapses_count;

    NeuronActivationPartition partitions[USHRT_MAX];
    size_t partitions_count = 0;
    CudaSynapse gpu_synapses[synapses_count];
    float efficacy[synapses_count];
    
    if(synapses_count == 0) {
        gpu.partitions_count = 0;
    } else {
        NeuronActivationPartition *currpartition = NULL;

        for(long i = 0; i < synapses_count; i++) {
            FiringRateModel__Synapse &synapse = synapses[i];
            if( (i % Threads_Per_Block == 0) || (synapse.toneuron != currpartition->toneuron) ) {
                if(currpartition)
                    currpartition++;
                else
                    currpartition = partitions;
                assert(currpartition - partitions < USHRT_MAX);

                currpartition->toneuron = synapse.toneuron;
                currpartition->offset = i % Threads_Per_Block;
                currpartition->len = 0;
            }
            currpartition->len++;

            CudaSynapse &gpu_synapse = gpu_synapses[i];
            gpu_synapse.fromneuron = synapse.fromneuron;
            gpu_synapse.partition = currpartition - partitions;
#if SYNAPSE_LEARN
            gpu_synapse.lrate = synapse.lrate;
#endif
            efficacy[i] = synapse.efficacy;
        }

        partitions_count = currpartition - partitions + 1;
        gpu.partitions_count = partitions_count;
    }

    uint sizeof_synapses = synapses_count * sizeof(CudaSynapse);
    size_t sizeof_partitions = partitions_count * sizeof(NeuronActivationPartition);
    size_t sizeof_activation = neurons_count * sizeof(float);
    size_t sizeof_efficacy = synapses_count * sizeof(float);

    sizeof_buffer =
        sizeof_synapses
        + sizeof_partitions
        + sizeof_activation
        + sizeof_efficacy;

    buffer = new unsigned char[sizeof_buffer];
    {
        uint offset = 0;

        memcpy(buffer + offset, gpu_synapses, sizeof_synapses);
        gpu.offsets.synapses = offset;
        offset += sizeof_synapses;

        memcpy(buffer + offset, partitions, sizeof_partitions);
        gpu.offsets.partitions = offset;
        offset += sizeof_partitions;

        memcpy(buffer + offset, neuronactivation, sizeof_activation);
        gpu.offsets.activations = offset;
        offset += sizeof_activation;

        memcpy(buffer + offset, efficacy, sizeof_efficacy);
        gpu.offsets.efficacies = offset;
        offset += sizeof_efficacy;
    }
}

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

void FiringRateModel_Cuda::alloc_update_buffers(AgentState *agents,
                                                long nagents,
                                                uint *input_offset,
                                                uint ninput,
                                                float **all_input,
                                                uint *output_offset,
                                                uint noutput,
                                                float **all_output) {
    if(d_all_input) {
        xcuda( cudaFreeHost(gpus) );
        xcuda( cudaFreeHost(buffers) );
        xcuda( cudaFreeHost(*all_input) );
        xcuda( cudaFreeHost(*all_output) );
        xcuda( cudaFree(d_gpus) );
        xcuda( cudaFree(d_buffers) );
        xcuda( cudaFree(d_all_input) );
        xcuda( cudaFree(d_all_output) );
    }

    sizeof_input = ninput * sizeof(float);
    sizeof_output = noutput * sizeof(float);
    uint sizeof_gpus = nagents * sizeof(GpuState);

    uint sizeof_buffers = 0;
    sizeof_shared = 0;
    for(long i = 0; i < nagents; i++) {
        AgentState &agent = agents[i];
        GpuState *gpu = &agent.model->gpu;

        uint sizeof_agent = uint((2 * gpu->neurons_count + Threads_Per_Block) * sizeof(float));
        sizeof_shared = max(sizeof_shared, sizeof_agent);

        sizeof_buffers += agent.model->sizeof_buffer;
    }

    xcuda( cudaMallocHost((void **)&gpus, sizeof_gpus) );
    xcuda( cudaMallocHost((void **)&buffers, sizeof_buffers) );
    xcuda( cudaMallocHost(all_input, sizeof_input) );
    xcuda( cudaMallocHost(all_output, sizeof_output) );

    xcuda( cudaMalloc((void**)&d_gpus, sizeof_gpus) );
    xcuda( cudaMalloc((void**)&d_buffers, sizeof_buffers) );
    xcuda( cudaMalloc((void**)&d_all_input, sizeof_input) );
    xcuda( cudaMalloc((void**)&d_all_output, sizeof_output) );

    uint buffers_offset = 0;
    for(long i = 0; i < nagents; i++) {
        AgentState &agent = agents[i];
        FiringRateModel_Cuda *model = agent.model;
        GpuState *gpu = &model->gpu;

        gpu->buffers.__main = d_buffers + buffers_offset;
        memcpy(buffers + buffers_offset, model->buffer, model->sizeof_buffer);
        buffers_offset += model->sizeof_buffer;

        gpu->buffers.input_activation = d_all_input + input_offset[i];
        gpu->buffers.output_activation = d_all_output + output_offset[i];

        gpus[i] = *gpu;
    }

    xcuda( cudaMemcpy(d_gpus, gpus, sizeof_gpus, cudaMemcpyHostToDevice) );
    xcuda( cudaMemcpy(d_buffers, buffers, sizeof_buffers, cudaMemcpyHostToDevice) );
}

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
