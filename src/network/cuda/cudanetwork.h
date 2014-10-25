#pragma once

#include "cudautil.h"
#include "network.h"

// Number of threads cannot exceed max value of ActivationPartition's offset and
// len fields. If they are of type uchar, then Threads_Per_Block must be < 256
#define Threads_Per_Block 32

// Use no more than 256 bytes of local memory for links
#define Max_Links_Per_Thread (256 / sizeof(CudaLink))
#define Max_Links (Max_Links_Per_Thread * Threads_Per_Block)


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

}
