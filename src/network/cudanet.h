#pragma once

#define NEURON_BIAS false
#define NEURON_TAU false
#define SYNAPSE_LEARN false

#define NEURON_ATTRS ( (NEURON_BIAS) || (NEURON_TAU) )

struct FiringRateModel__Neuron {
#if NEURON_BIAS
    float bias;
#endif
#if NEURON_TAU
    float tau;
#endif
    long  startsynapses;
    long  endsynapses;
};

struct FiringRateModel__NeuronAttrs
{
#if NEURON_BIAS
    float bias;
#endif
#if NEURON_TAU
    float tau;
#endif
};

struct FiringRateModel__Synapse
{
    float efficacy;   // > 0 for excitatory, < 0 for inhibitory
#if SYNAPSE_LEARN
    float lrate;
#endif
    short fromneuron;
    short toneuron;
};

typedef void *Identity;

struct FiringRateModel_Cuda {

    struct AgentState {
        Identity id;
        FiringRateModel_Cuda *model;
        float *neuronactivation;
        float *newneuronactivation;
    };

    static void alloc_update_buffers(AgentState *agents,
                                     long nagents,
                                     uint *input_offset,
                                     uint ninput,
                                     float **all_input,
                                     uint *output_offset,
                                     uint noutput,
                                     float **all_output);
    static void update_all(AgentState *agents,
                           long nagents,
                           float *all_input,
                           float *all_output);

    struct Neuron {
#if NEURON_BIAS
        float bias;
#endif
#if NEURON_TAU
        float tau;
#endif
    };
    
    struct Synapse {
        short fromneuron;
        unsigned short partition;
#if SYNAPSE_LEARN
        float lrate;
#endif
    };

    struct NeuronActivationPartition {
        short toneuron;
        short offset;
        short len;
        short __padding;
    };

    FiringRateModel_Cuda();
    ~FiringRateModel_Cuda();

    void init(FiringRateModel__Neuron *neurons,
              short neurons_count, short input_neurons_count, short output_neurons_count,
              float *neuronactivation,
              FiringRateModel__Synapse *synapses,
              long synapses_count,
              float logistic_slope,
              float decay_rate,
              float max_weight);

    unsigned char *buffer;
    size_t sizeof_buffer;

    struct GpuState {
        short neurons_count;
        short input_neurons_count;
        short output_neurons_count;
        unsigned short partitions_count;
        long synapses_count;
        float logistic_slope;
        float decay_rate;
        float max_weight;

        struct {
            uint neurons;
            uint synapses;
            uint partitions;
            uint activations;
            uint efficacies;
        } offsets;

        struct {
            unsigned char *__main;

            float *input_activation;
            float *output_activation;
        } buffers;

#ifdef DEVICE_CODE

        inline __device__ Neuron *neurons() {
            return (Neuron *)(buffers.__main + offsets.neurons);
        }

        inline __device__ Synapse *synapses() {
            return (Synapse *)(buffers.__main + offsets.synapses);
        }

        inline __device__ NeuronActivationPartition *partitions() {
            return (NeuronActivationPartition *)(buffers.__main + offsets.partitions);
        }

        inline __device__ float *activations() {
            return (float *)(buffers.__main + offsets.activations);
        }

        inline __device__ float *efficacies() {
            return (float *)(buffers.__main + offsets.efficacies);
        }
#endif
    } gpu;
};
