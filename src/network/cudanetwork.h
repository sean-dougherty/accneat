#pragma once

#include "network.h"

namespace NEAT {

    struct CudaLink {
        node_size_t in_node_index;
        real_t weight;
        link_size_t partition;
    };

    struct ActivationPartition {
        node_size_t out_node_index;
        ushort offset;
        ushort len;
        short __padding;
    };

    struct ActivateParms {
        uint flush_step;
    };

    struct Offsets {
        struct {
            uint activation;
            uint links;
            uint partitions;
        } main;
        struct {
            uint activate_parms;
            uint activation;
        } input;
        struct {
            uint activation;
        } output;
    };

    struct Lens {
        uint main;
        uint input;
        uint output;

        friend Lens operator+(const Lens &a, const Lens &b) {
            Lens c;
            c.main = a.main + b.main;
            c.input = a.input + b.input;
            c.output = a.output + b.output;
            return c;
        }

        friend bool operator>(const Lens &a, const Lens &b) {
            return a.main > b.main
                || a.input > b.input
                || a.output > b.output;
        }
    };

    struct RawBuffers {
        uchar *main;
        uchar *input;
        uchar *output;
    };

    struct CudaNetDims : public NetDims {
        link_size_t npartitions;
    };

    struct GpuState {
        CudaNetDims dims;
        Offsets offsets;
    };

    class CudaNetwork : public Network {
    private:
        class CudaNetworkBatch *batch;
        RawBuffers bufs;
        CudaNetDims dims;
        Offsets offsets;

    public:
        CudaNetwork() {}
        virtual ~CudaNetwork() {}

        void set_batch(class CudaNetworkBatch *batch_) {batch = batch_;}
        void get_gpu(__out GpuState &gpu);
        void set_bufs(const RawBuffers &bufs);

        virtual void configure(const NetDims &counts,
                               NetNode *nodes,
                               NetLink *links);
		virtual void load_sensors(const std::vector<real_t> &sensvals,
                                  bool clear_noninput);
        virtual real_t get_output(size_t index);
    };

    class CudaNetworkBatch {
        uint nnets;
        CudaNetDims template_dims;
        RawBuffers h_bufs;
        RawBuffers d_bufs;
        GpuState *h_gpu_states;
        GpuState *d_gpu_states;
        Offsets offsets;
        Lens capacity;
        Lens lens;
        uint step;

    public:
        CudaNetworkBatch(uint nnets);
        ~CudaNetworkBatch();

        uint get_step() {return step;}

        void configure_net(__in CudaNetDims &dims,
                           __out RawBuffers &bufs,
                           __out Offsets &offsets);

        void configure_device(CudaNetwork **nets,
                              uint nnets);

        void activate();
    };


#if false
    class CudaNetworkBatch {
        struct {
            uint buflen;
            uchar *buffer;
            uchar *d_buffer;

            uint offset_activations;
            uint offset_parms;
        } update_input;

        struct {
            uint buflen;
            uchar *buffer;
            uchar *d_buffer;
        } update_output;

        struct {
            CudaNetwork::GpuState *gpus;
            CudaNetwork::GpuState *d_gpus;

            uchar *buffers;
            uchar *d_buffers;
        } config;

        uint sizeof_shared;
    public:
        CudaNetworkBatch(CudaNetwork **nets, size_t nnets);
        ~CudaNetworkBatch();

        void activate();
    };
#endif
}

#if false
struct FiringRateModel__Neuron {
    long  startsynapses;
    long  endsynapses;
};

struct FiringRateModel__Synapse {
    float efficacy;   // > 0 for excitatory, < 0 for inhibitory
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

    unsigned char *buffer;
    size_t sizeof_buffer;

    struct GpuState {
        short neurons_count;
        short input_neurons_count;
        short output_neurons_count;
        ushort partitions_count;
        long synapses_count;

        struct {
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

        inline __device__ CudaSynapse *synapses() {
            return (CudaSynapse *)(buffers.__main + offsets.synapses);
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
#endif
