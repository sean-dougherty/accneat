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
        bool clear_noninput;
        bool enabled;
        short __padding; // This precedes, activations, which must be properly aligned for the
                         // GPU to read them properly!!!
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

        Lens &operator+=(const Lens &a) {
            main += a.main;
            input += a.input;
            output += a.output;
            return *this;
        }

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
        friend class CudaNetworkBatch;

        CudaNetDims dims;
        std::vector<CudaLink> gpu_links;
        std::vector<ActivationPartition> partitions;

        RawBuffers bufs;
        Offsets offsets;

    public:
        CudaNetwork() {}
        virtual ~CudaNetwork() {}

        void set_bufs(const RawBuffers &bufs, const Offsets &offsets);

        void set_clear_noninput(bool val);

        void disable();
        bool is_enabled();

        virtual void configure(const NetDims &counts,
                               NetNode *nodes,
                               NetLink *links);

        virtual NetDims get_dims() { return dims; }

		virtual void load_sensors(const std::vector<real_t> &sensvals,
                                  bool clear_noninput);

        virtual real_t get_output(size_t index);
    };

    class CudaNetworkBatch {
        uint nnets;
        GpuState *h_gpu_states;
        GpuState *d_gpu_states;

        RawBuffers h_bufs;
        RawBuffers d_bufs;
        Offsets offsets;
        Lens capacity;
        Lens lens;
        uint sizeof_shared;

    public:
        CudaNetworkBatch(uint nnets);
        ~CudaNetworkBatch();

        void configure(CudaNetwork **nets, uint nnets);

        void activate(uint ncycles);
        std::vector<real_t> &get_activations(CudaNetwork *net, __out std::vector<real_t> &result);
    };

    void test_sum_partition();
}
