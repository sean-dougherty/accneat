#pragma once

#include "network.h"
#include "networkexecutor.h"
#include "cudautil.h"
#include <assert.h>

namespace NEAT {

// Number of threads cannot exceed max value of ActivationPartition's offset and
// len fields. If they are of type uchar, then Threads_Per_Block must be < 256
#define Threads_Per_Block 32

// Use no more than 256 bytes of local memory for links
#define Max_Links_Per_Thread (256 / sizeof(CudaLink))
#define Max_Links (Max_Links_Per_Thread * Threads_Per_Block)

#define __net_eval_decl __host__ __device__

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
            uint states;
        } gpu_states;
        struct {
            uint evals;
        } output;
    };

    //---
    //--- CLASS Lens
    //---
    struct Lens {
        uint main;
        uint gpu_states;
        uint output;
    };

    //---
    //--- CLASS RawBuffers
    //---
    struct RawBuffers {
        uchar *main;
        uchar *gpu_states;
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

    static __net_eval_decl CudaLink *links(const RawBuffers &bufs,
                                           const Offsets &offs) {
        return (CudaLink *)(bufs.main + offs.main.links);
    }

    static __net_eval_decl ActivationPartition *partitions(const RawBuffers &bufs,
                                                           const Offsets &offs) {
        return (ActivationPartition *)(bufs.main + offs.main.partitions);
    }

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

        const CudaNetDims &get_cuda_dims() {return dims;}
        const Offsets &get_offsets() {return offsets;}
        virtual NetDims get_dims() { return dims; }

        void configure_batch(const RawBuffers &bufs_,
                             const Offsets &offsets_) {
            bufs = bufs_;
            offsets = offsets_;

            memcpy( NEAT::links(bufs, offsets),
                    gpu_links.data(),
                    sizeof(CudaLink) * gpu_links.size() );
            memcpy( NEAT::partitions(bufs, offsets),
                    partitions.data(),
                    sizeof(ActivationPartition) * partitions.size() );
        }

        virtual void configure(const NetDims &dims_,
                               NetNode *nodes,
                               NetLink *links) {
            static_cast<NetDims &>(dims) = dims_;

            require(dims.nlinks < Max_Links);

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

    };

    template<typename Evaluator>
    static __global__
    void cudanetwork_activate(const typename Evaluator::Config *config,
                              RawBuffers bufs,
                              uint cycles) {
        Evaluator eval(config);
    }

    template<typename Evaluator>
    class CudaNetworkBatch {
        typedef typename Evaluator::Config Config;
        int device;
        uint nnets;
        Offsets offsets;
        Lens capacity;
        Lens lens;
        uint sizeof_shared;

        Config *d_config;
        RawBuffers h_bufs;
        RawBuffers d_bufs;

        void configure(CudaNetwork **nets, size_t nnets);

    public:
        CudaNetworkBatch(int device_)
            : device(device_) {

            cudaSetDevice(device);

            memset(&offsets, 0, sizeof(offsets));
            memset(&capacity, 0, sizeof(capacity));
            memset(&lens, 0, sizeof(lens));
            sizeof_shared = 0;

            d_config = NULL;
            memset(&h_bufs, 0, sizeof(h_bufs));
            memset(&d_bufs, 0, sizeof(d_bufs));
        }

        ~CudaNetworkBatch() {
            free_dev(d_config);

            free_host(h_bufs.main);
            free_dev(d_bufs.main);
            free_host(h_bufs.gpu_states);
            free_dev(d_bufs.gpu_states);
            free_host(h_bufs.output);
            free_dev(d_bufs.output);
        }

        void configure(const Config *config,
                       size_t len) {
            cudaSetDevice(device);

            free_dev(d_config);
            d_config = (Config *)alloc_dev(len);
            xcuda( cudaMemcpy(d_config, config, len, cudaMemcpyHostToDevice) );
        }

        void activate(CudaNetwork **nets,
                      OrganismEvaluation *results,
                      size_t nnets,
                      size_t ncycles) {
            cudaSetDevice(device);

            configure(nets, nnets);

            cudanetwork_activate<Evaluator><<<nnets, Threads_Per_Block, sizeof_shared>>>(d_config, d_bufs, ncycles);
            
            xcuda( cudaMemcpy(h_bufs.output,
                              d_bufs.output,
                              lens.output,
                              cudaMemcpyDeviceToHost) );
        }
    };

    template<typename Evaluator>
    void CudaNetworkBatch<Evaluator>::configure(CudaNetwork **nets, uint nnets) {
        cudaSetDevice(device);

        memset(&lens, 0, sizeof(lens));
        sizeof_shared = 0;
            
        Offsets nets_offs[nnets];

        for(uint i = 0; i < nnets; i++) {
            CudaNetwork &net = *nets[i];
            const CudaNetDims &dims = net.get_cuda_dims();

            Lens net_lens;
            Offsets &net_offs = nets_offs[i];
            uint net_sizeof_shared =
                (2 * sizeof(real_t) * dims.nnodes.all)
                + (sizeof(real_t) * Threads_Per_Block);

            //main buffer
            {
                uint sizeof_links = sizeof(CudaLink) * dims.nlinks;
                uint sizeof_partitions = sizeof(ActivationPartition) * dims.npartitions;

                net_lens.main = sizeof_links + sizeof_partitions;

                net_offs.main.links = this->lens.main;
                net_offs.main.partitions = net_offs.main.links + sizeof_links;

                lens.main += net_lens.main;
            }

            //gpu_states buffer
            {
                uint sizeof_state = sizeof(GpuState);

                net_lens.gpu_states = sizeof_state;

                net_offs.gpu_states.states = this->lens.gpu_states;

                lens.gpu_states += net_lens.gpu_states;
            }

            //output buffer
            {
                uint sizeof_evals = 
                    sizeof(OrganismEvaluation);

                net_lens.output = sizeof_evals;

                net_offs.output.evals = this->lens.output;

                lens.output += net_lens.output;
            }

            sizeof_shared = max(sizeof_shared, net_sizeof_shared);
        }            

        if(lens.main > capacity.main) {
            uint newlen = uint(lens.main * 1.4);
            p("alloc main: " << newlen);
            grow_buffers(h_bufs.main, d_bufs.main, capacity.main, newlen);
        }
        if(lens.gpu_states > capacity.gpu_states) {
            uint newlen = uint(lens.gpu_states * 1.4);
            p("alloc gpu_states: " << newlen);
            grow_buffers(h_bufs.gpu_states, d_bufs.gpu_states, capacity.gpu_states, newlen);
        }
        if(lens.output > capacity.output) {
            uint newlen = uint(lens.output);
            p("alloc output: " << newlen);
            assert(capacity.output == 0); // should only alloc once
            grow_buffers(h_bufs.output, d_bufs.output, capacity.output, newlen);
        }
            
        for(uint i = 0; i < nnets; i++) {
            CudaNetwork *net = nets[i];
            net->configure_batch(h_bufs, nets_offs[i]);

            GpuState &gpu = ((GpuState *)h_bufs.gpu_states)[i];
            gpu.dims = net->get_cuda_dims();
            gpu.offsets = net->get_offsets();
        }

        xcuda( cudaMemcpy(d_bufs.main, h_bufs.main, lens.main, cudaMemcpyHostToDevice) );
        xcuda( cudaMemcpy(d_bufs.gpu_states, h_bufs.gpu_states, lens.gpu_states, cudaMemcpyHostToDevice) );
    }

    //---
    //--- CLASS CudaNetworkExecutor
    //---
    template<typename Evaluator>
    class CudaNetworkExecutor : public NetworkExecutor<Evaluator> {
        std::vector<class CudaNetworkBatch<Evaluator> *> batches;
    public:
        CudaNetworkExecutor() {
            int ndevices;
            xcuda( cudaGetDeviceCount(&ndevices) );
            errif(ndevices == 0, "No Cuda devices found!");

            batches.resize(ndevices);
            for(int i = 0; i < ndevices; i++) {
                batches[i] = new CudaNetworkBatch<Evaluator>(i);
            }
        }

        virtual ~CudaNetworkExecutor() {
            for(size_t i = 0; i < batches.size(); i++) {
                delete batches[i];
            }
        }

        virtual void configure(const typename Evaluator::Config *config,
                               size_t len) {
            for(size_t i = 0; i < batches.size(); i++) {
                batches[i]->configure(config, len);
            }
        }

        virtual void execute(class Network **nets_,
                             OrganismEvaluation *results,
                             size_t nnets) {
            CudaNetwork **nets = (CudaNetwork **)nets_;
            size_t nbatches = batches.size();
            uint batch_size = nnets / nbatches;

#pragma omp parallel for
            for(size_t ibatch = 0; ibatch < nbatches; ibatch++) {
                size_t inet = ibatch * batch_size;
                size_t n = batch_size;
                if(ibatch == nbatches - 1)
                    n += nnets % batch_size;

                batches[ibatch]->activate(nets + inet,
                                          results + inet,
                                          n,
                                          NACTIVATES_PER_INPUT);
            }
            
        }
        
    };

    template<typename Evaluator>
    inline NetworkExecutor<Evaluator> *NetworkExecutor<Evaluator>::create() {
        return new CudaNetworkExecutor<Evaluator>();
    }
}
