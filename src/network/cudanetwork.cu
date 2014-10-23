#define DEVICE_CODE

#include <iostream>
#include <vector>

#include "network.h"

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

// Number of threads cannot exceed max value of ActivationPartition's offset and
// len fields. If they are of type uchar, then Threads_Per_Block must be < 256
#define Threads_Per_Block 32

// Use no more than 256 bytes of local memory for links
#define Max_Links_Per_Thread (256 / sizeof(CudaLink))
#define Max_Links (Max_Links_Per_Thread * Threads_Per_Block)

#define xcuda(stmt) {                                                   \
        cudaError_t err = stmt;                                         \
        if (err != cudaSuccess) {                                       \
            std::cerr << __FILE__ << ":" << __LINE__ << ": Failed to run " << #stmt << ". Reason: " << cudaGetErrorString(err) << std::endl; \
            abort();                                                    \
        }                                                               \
    }

namespace NEAT {
/*
    __global__ void activate(GpuState *states,
                             RawBuffers bufs,
                             CudaBatchSensorsDims sensor_dims,
                             uint ncycles);
*/

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
    static void free_host(__inout uchar *&buf, bool tolerate_shutdown = false) {
        if(buf) {
            cudaError_t err = cudaFreeHost(buf);
            if( (err == cudaSuccess)
                || (tolerate_shutdown && (err == cudaErrorCudartUnloading)) ) {
                buf = 0;
            } else {
                std::cerr << "Failed freeing cuda host buffer" << std::endl;
                abort();
            }
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

    __dh_util CudaLink *links(const RawBuffers &bufs,
                              const Offsets &offs) {
        return (CudaLink *)(bufs.main + offs.main.links);
    }

    __dh_util ActivationPartition *partitions(const RawBuffers &bufs,
                                              const Offsets &offs) {
        return (ActivationPartition *)(bufs.main + offs.main.partitions);
    }

#undef __dh_util

//--------------------------------------------------------------------------------
//---
//--- CLASS CudaNetwork
//---
//--------------------------------------------------------------------------------
    void CudaNetwork::configure_batch(const RawBuffers &bufs_,
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

    void CudaNetwork::configure(const NetDims &dims_,
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

#if false

//--------------------------------------------------------------------------------
//---
//--- CLASS CudaNetworkBatch
//---
//--------------------------------------------------------------------------------
    CudaNetworkBatch::CudaNetworkBatch(int device_, uint nnets_)
        : device(device_), nnets(nnets_) {

        cudaSetDevice(device);

        memset(&h_bufs, 0, sizeof(h_bufs));
        memset(&d_bufs, 0, sizeof(d_bufs));
        memset(&offsets, 0, sizeof(offsets));
        memset(&capacity, 0, sizeof(capacity));
        memset(&lens, 0, sizeof(lens));

        h_gpu_states = (GpuState *)alloc_host(sizeof(GpuState) * nnets);
        d_gpu_states = (GpuState *)alloc_dev(sizeof(GpuState) * nnets);
    }

    CudaNetworkBatch::~CudaNetworkBatch() {
        cudaSetDevice(device);

        free_host((uchar *&)h_gpu_states);
        free_dev((uchar *&)d_gpu_states);

        free_host(h_bufs.main);
        // the host input buffer is in the sensors object
        free_host(h_bufs.output);

        free_dev(d_bufs.main);
        free_dev(d_bufs.input);
        free_dev(d_bufs.output);
    }

    void CudaNetworkBatch::configure(CudaBatchSensors *batch_sensors,
                                     CudaNetwork **nets,
                                     uint nnets) {
        assert(nnets = this->nnets);

        cudaSetDevice(device);

        sensor_dims = batch_sensors->get_dims();
        memset(&lens, 0, sizeof(lens));
        sizeof_shared = 0;
        istep_output = 0;

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
                uint sizeof_links = sizeof(CudaLink) * dims.nlinks;
                uint sizeof_partitions = sizeof(ActivationPartition) * dims.npartitions;

                net_lens.main = sizeof_links + sizeof_partitions;

                net_offs.main.links = this->lens.main;
                net_offs.main.partitions = net_offs.main.links + sizeof_links;

                lens.main += net_lens.main;
            }

            //input buffer
            {
                uint sizeof_step_parms = StepParms::sizeof_buf(sensor_dims);

                net_lens.input = sizeof_step_parms;

                //If all nets share the same input.
                {
                    assert(sizeof_step_parms == batch_sensors->sizeof_buf());
                    net_offs.input.step_parms = 0;
                    lens.input = max(lens.input, net_lens.input);
                }
            }

            //output buffer
            {
                uint sizeof_activation = 
                    sizeof(real_t) * dims.nnodes.output * sensor_dims.nsteps;

                net_lens.output = sizeof_activation;

                net_offs.output.activation = this->lens.output;
                // gpu requires proper alignment
                assert(net_offs.output.activation % sizeof(real_t) == 0);

                lens.output += net_lens.output;
            }

            sizeof_shared = max(sizeof_shared, net_sizeof_shared);
            
            net.offsets = net_offs;
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
            d_bufs.input = alloc_dev(newlen);
            capacity.input = newlen;
        }
        if(lens.output > capacity.output) {
            uint newlen = uint(lens.output);
            p("alloc output: " << newlen);
            assert(capacity.output == 0); // should only alloc once
            grow_buffers(h_bufs.output, d_bufs.output, capacity.output, newlen);
        }

        for(uint i = 0; i < nnets; i++) {
            CudaNetwork *net = nets[i];
            net->configure_batch(h_bufs, nets_offs[i], &istep_output);

            GpuState &gpu = h_gpu_states[i];
            gpu.dims = net->dims;
            gpu.offsets = net->offsets;
        }

        xcuda( cudaMemcpy(d_bufs.input, batch_sensors->get_h_buf(), lens.input, cudaMemcpyHostToDevice) );
        xcuda( cudaMemcpy(d_gpu_states, h_gpu_states, sizeof(GpuState) * nnets, cudaMemcpyHostToDevice) );
        xcuda( cudaMemcpy(d_bufs.main, h_bufs.main, lens.main, cudaMemcpyHostToDevice) );
    }

    void CudaNetworkBatch::activate(uint ncycles) {
        NEAT::activate<<<nnets, Threads_Per_Block, sizeof_shared>>>(d_gpu_states,
                                                                    d_bufs,
                                                                    sensor_dims,
                                                                    ncycles);
        
        xcuda( cudaMemcpy(h_bufs.output,
                          d_bufs.output,
                          lens.output,
                          cudaMemcpyDeviceToHost) );
    }

    void CudaNetworkBatch::set_output_step(size_t istep) {
        istep_output = istep;
    }

//--------------------------------------------------------------------------------
//---
//--- CLASS CudaBatchSensors
//---
//--------------------------------------------------------------------------------
    CudaBatchSensors::CudaBatchSensors(const CudaBatchSensorsDims &dims_)
        : dims(dims_) {

        h_buf = alloc_host(sizeof_buf());
    }

    CudaBatchSensors::~CudaBatchSensors() {
        free_host(h_buf, true);
    }

    void CudaBatchSensors::configure_step(size_t istep,
                                          const std::vector<real_t> &values,
                                          bool clear_noninput) {
        assert(values.size() == dims.nsensors);
        if(istep == 0) {
            require(clear_noninput);
        }

        StepParms *parms = StepParms::get(h_buf, dims, istep);

        parms->clear_noninput = clear_noninput;
        for(size_t i = 0; i < dims.nsensors; i++)
            parms->activations[i] = values[i];
    }

//--------------------------------------------------------------------------------
//---
//--- GPU KERNEL CODE
//---
//--------------------------------------------------------------------------------
    __device__ void sum_partition(float *x, int i, int n, float *result) {
        int stride = __popc(n) == 1 ? n >> 1 : 1 << 31 - __clz(n);

        if( (stride > 0) && (i + stride < n) ) {
            x[i] += x[i + stride];
        }
      
        __syncthreads();

        stride >>= 1;
        // max_stride necessary to keep all threads from all partitions in sync.
        for(int max_stride = Threads_Per_Block >> 1; max_stride > 0; stride >>= 1, max_stride >>= 1) {
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

    __global__ void test_sum_partition_kernel(float *x, int n, float *result) {
        uint tid = threadIdx.x;
        __shared__ float shx[Threads_Per_Block];
        *result = 0;
        int i;
        if(tid < n) {
            shx[tid] = x[tid];
            i = tid;
        } else {
            i = 1; n = 0;
        }
        sum_partition(shx, i, n, result);
    }

    inline __device__ real_t fsigmoid(real_t activesum,real_t slope,real_t constant) {
        //NON-SHIFTED STEEPENED
        return (1/(1+(exp(-(slope*activesum))))); //Compressed
    }

    __global__ void activate(GpuState *states,
                             RawBuffers bufs,
                             CudaBatchSensorsDims sensor_dims,
                             uint ncycles) {
        // to print sensors:
        // p *(@global float * @local)(real_t *foo)@sensor_dims.nsensors

        GpuState state = states[blockIdx.x];
        uint tid = threadIdx.x;

        //---
        //--- Config shared memory
        //---
        extern __shared__ char __shared_buf[];

        // in cuda-gdb: print *((@shared float*)activation + i)
        //              print *((@shared float*)newactivation)@6
        real_t *activation = (real_t *)__shared_buf;
        real_t *newactivation = activation + state.dims.nnodes.all;
        real_t *partial_activation = newactivation + state.dims.nnodes.all;


        //---
        //--- Cache link/partitions in local memory.
        //---
        const int ncycle_its = 1 + (state.dims.nlinks - 1) / Threads_Per_Block;

        CudaLink local_links[Max_Links_Per_Thread];
        ActivationPartition local_partitions[Max_Links_Per_Thread];
        for(uint ilink = tid, it = 0; it < ncycle_its; ilink += Threads_Per_Block, it++) {
            CudaLink &link = local_links[it];
            ActivationPartition &p = local_partitions[it];
            if(ilink < state.dims.nlinks) {
                link = links(bufs, state.offsets)[ilink];
                p = partitions(bufs, state.offsets)[local_links[it].partition];
            }
        }


        //---
        //--- Process all batch steps
        //---
        for(size_t istep = 0; istep < sensor_dims.nsteps; istep++) {
            StepParms *parms = step_parms(bufs, state.offsets, sensor_dims, istep);

            //---
            //--- Load step activations
            //---
            for(uint inode = tid; inode < state.dims.nnodes.all; inode += Threads_Per_Block) {
                if(inode < state.dims.nnodes.input) {
                    //---
                    //--- Bias/Sensor node
                    //---
                    const uint nbias = state.dims.nnodes.bias;
                    if(inode < nbias) {
                        activation[inode] = 1.0;
                    } else {
                        activation[inode] =
                            parms->activations[inode - nbias];
                    }
                    newactivation[inode] = activation[inode];
                } else {
                    //---
                    //--- Output/Hidden node
                    //---
                    if( parms->clear_noninput ) {
                        activation[inode] = 0.0;
                    }
                }
            }
            __syncthreads();

            //---
            //--- For each cycle of this step.
            //---
            for(uint icycle = 0; icycle < ncycles; icycle++) {

                //---
                //--- Reset new activation noninput
                //---
                for(uint inode = tid + state.dims.nnodes.input;
                    inode < state.dims.nnodes.all;
                    inode += Threads_Per_Block) {

                    newactivation[inode] = 0.0;
                }

                //---
                //--- Compute new activation sums
                //---
                for(uint ilink = tid, it = 0; it < ncycle_its; ilink += Threads_Per_Block, it++) {
                    float *partition_x;
                    int partition_i;
                    int partition_n;
                    float *result;

                    if(ilink < state.dims.nlinks) {
                        CudaLink &link = local_links[it];
                        partial_activation[tid] = link.weight * activation[link.in_node_index];

                        ActivationPartition &p = local_partitions[it];
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
                }

                //---
                //--- Compute new activations from sums
                //---
                for(uint inode = tid + state.dims.nnodes.input;
                    inode < state.dims.nnodes.all;
                    inode += Threads_Per_Block) {

                    newactivation[inode] = fsigmoid(newactivation[inode],
                                                    4.924273,
                                                    2.4621365);
                }
                __syncthreads();

                //---
                //--- "activation" now the current state.
                //---
                {
                    float *swap = newactivation;
                    newactivation = activation;
                    activation = swap;
                }
            } // end of cycle

            //---
            //--- Save step output to global memory
            //---
            for(uint i = tid; i < state.dims.nnodes.output; i += Threads_Per_Block) {
                output_activations(bufs, state.offsets, state.dims, istep)[i] = 
                    activation[state.dims.nnodes.input + i];
            }
        } // end of step
    }

    void test_sum_partition() {
        for(size_t n = 1; n <= Threads_Per_Block; n++) {
            real_t x[n];
            size_t sizeof_x = sizeof(real_t) * n;

            real_t expected = 0.0;
            for(size_t i = 0; i < n; i++) {
                //x[i] = real_t(i) + 1;
                x[i] = drand48();
                expected += x[i];
            }

            real_t actual = -100;

            real_t *d_x = (real_t *)alloc_dev(sizeof_x);
            xcuda( cudaMemcpy(d_x,
                              x,
                              sizeof_x,
                              cudaMemcpyHostToDevice) );

            real_t *d_actual = (real_t *)alloc_dev(sizeof(real_t));

            NEAT::test_sum_partition_kernel<<<1, Threads_Per_Block>>>(d_x, n, d_actual);
            
            xcuda( cudaMemcpy(&actual,
                              d_actual,
                              sizeof(real_t),
                              cudaMemcpyDeviceToHost) );

            if( fabs(expected - actual) / expected >= 0.05 ) {
                std::cout << "n=" << n << ", Expected=" << expected << ", Actual=" << actual << std::endl;
            }
        }
            
        exit(0);
    }
#endif // false

} // namespace NEAT

