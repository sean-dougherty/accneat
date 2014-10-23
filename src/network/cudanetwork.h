#if false

#pragma once

#ifndef DEVICE_CODE
#define __host__
#define __device__
#endif

#define __dh_util static inline __device__ __host__


namespace NEAT {

    //---
    //--- CLASS CudaLink
    //---
    struct CudaLink {
        link_size_t partition;
        node_size_t in_node_index;
        real_t weight;
    };

    //---
    //--- CLASS ActivationPartition
    //---
    struct ActivationPartition {
        node_size_t out_node_index;
        uchar offset;
        uchar len;
    };

    //---
    //--- CLASS Offsets
    //---
    struct Offsets {
        struct {
            uint links;
            uint partitions;
        } main;
        struct {
            uint step_parms;
        } input;
        struct {
            uint activation;
        } output;
    };

    //---
    //--- CLASS Lens
    //---
    struct Lens {
        uint main;
        uint input;
        uint output;
    };

    //---
    //--- CLASS RawBuffers
    //---
    struct RawBuffers {
        uchar *main;
        uchar *input;
        uchar *output;
    };

    //---
    //--- CLASS CudaNetDims
    //---
    struct CudaNetDims : public NetDims {
        link_size_t npartitions;
    };

    //---
    //--- CLASS GpuState
    //---
    struct GpuState {
        CudaNetDims dims;
        Offsets offsets;
    };

    //---
    //--- CLASS CudaNetwork
    //---
    class CudaNetwork : public Network {
    private:
        CudaNetDims dims;
        std::vector<CudaLink> gpu_links;
        std::vector<ActivationPartition> partitions;

        RawBuffers bufs;
        Offsets offsets;

    public:
        CudaNetwork() {}
        virtual ~CudaNetwork() {}

        void configure_batch(const RawBuffers &bufs,
                             const Offsets &offsets);

        virtual void configure(const NetDims &counts,
                               NetNode *nodes,
                               NetLink *links);

        virtual NetDims get_dims() { return dims; }
    };

    //---
    //--- CLASS CudaNetworkBatch
    //---
    template<typename Evaluator>
    class CudaNetworkBatch {
        int device;
        uint nnets;
        const typename Evaluator::Config *d_config;
        GpuState *h_gpu_states;
        GpuState *d_gpu_states;

        RawBuffers h_bufs;
        RawBuffers d_bufs;
        Offsets offsets;
        Lens capacity;
        Lens lens;
        uint sizeof_shared;

    public:
        CudaNetworkBatch(int device_, uint nnets_)
            : device(device_)
            , nnets(nnets_)
            , d_config(NULL) {

            cudaSetDevice(device);

            memset(&h_bufs, 0, sizeof(h_bufs));
            memset(&d_bufs, 0, sizeof(d_bufs));
            memset(&offsets, 0, sizeof(offsets));
            memset(&capacity, 0, sizeof(capacity));
            memset(&lens, 0, sizeof(lens));

            h_gpu_states = (GpuState *)alloc_host(sizeof(GpuState) * nnets);
            d_gpu_states = (GpuState *)alloc_dev(sizeof(GpuState) * nnets);
        }
        ~CudaNetworkBatch();

        void configure(const typename Evaluator::Config *config);

        void configure_nets(CudaNetwork **nets,
                            uint nnets);

        void activate(uint ncycles);
    };

#ifdef DEVICE_CODE
    template<typename Evaluator>
    __global__ void foo() {
        Evaluator eval(NULL);
    }
#endif

/*
    //---
    //--- CLASS CudaNetworkExecutor
    //---
    template<typename Evaluator>
    class CudaNetworkExecutor : public NetworkExecutor<Evaluator> {
    public:
        const typename Evaluator::Config *config = nullptr;

        virtual ~CudaNetworkExecutor() {
            delete config;
        }

        virtual void configure(const typename Evaluator::Config *config_,
                               size_t len) override {
            void *buf = malloc(len);
            memcpy(buf, config_, len);
            config = (const typename Evaluator::Config *)buf;
        }

        virtual void execute(class Network **nets_,
                             OrganismEvaluation *results,
                             size_t nnets) override {

            CudaNetwork **nets = (CudaNetwork **)nets_;
            size_t nsensors = nets[0]->get_dims().nnodes.sensor;

#pragma omp parallel for
            for(size_t inet = 0; inet < nnets; inet++) {
                CudaNetwork *net = nets[inet];
                Evaluator eval{config};

                for(size_t istep = 0; !eval.complete(istep); istep++) {
                    if(eval.clear_noninput(istep)) {
                        net->clear_noninput();
                    }
                    for(size_t isensor = 0; isensor < nsensors; isensor++) {
                        net->load_sensor(isensor, eval.get_sensor(istep, isensor));
                    }
                    net->activate(NACTIVATES_PER_INPUT);
                    eval.evaluate(istep, net->get_outputs());
                }

                results[inet] = eval.result();
            }
        }
        
    };
*/

#if false
    void test_sum_partition();
#endif // false
}

#endif
