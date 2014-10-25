#pragma once

#include "cudautil.h"
#include "cudanetwork.h"
#include "cudanetworkkernel.h"

namespace NEAT {

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

}
