#pragma once

namespace NEAT {

//--------------------------------------------------------------------------------
//---
//--- GPU KERNEL CODE
//---
//--------------------------------------------------------------------------------
    inline __device__ void sum_partition(float *x, int i, int n, float *result) {
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

    inline __device__ real_t fsigmoid(real_t activesum,real_t slope,real_t constant) {
        //NON-SHIFTED STEEPENED
        return (1/(1+(exp(-(slope*activesum))))); //Compressed
    }

    template<typename Evaluator>
    static __global__
    void cudanetwork_activate(const typename Evaluator::Config *config,
                              RawBuffers bufs,
                              uint ncycles) {
        Evaluator eval(config);

        // to print sensors:
        // p *(@global float * @local)(real_t *foo)@sensor_dims.nsensors

        GpuState state = ((GpuState *)bufs.gpu_states)[blockIdx.x];
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
        for(size_t istep = 0; !eval.complete(istep); istep++) {
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
                        //Bias node, so just set to 1.0
                        activation[inode] = 1.0;
                    } else {
                        //Sensor 
                        activation[inode] =
                            eval.get_sensor(istep, inode - nbias);
                    }
                    newactivation[inode] = activation[inode];
                } else {
                    //---
                    //--- Output/Hidden node
                    //---
                    if( eval.clear_noninput(istep) ) {
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

            } //for each cycle

            //---
            //--- Evaluate output for this step. For now, only use one thread.
            //--- In the future, may want to parallelize this. At the moment,
            //--- the number of outputs is typically very small, so the complexity
            //--- of a parallel computation doesn't seem worth it.
            //---
            if(tid == 0) {
                eval.evaluate(istep, activation + state.dims.nnodes.input);
            }

        } //for each step

        //---
        //--- Save evaluation to output buffer in global memory
        //---
        if(tid == 0) {
            ((OrganismEvaluation *)bufs.output)[blockIdx.x] = eval.result();
        }
    } // kernel

}
