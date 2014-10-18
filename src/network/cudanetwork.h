#pragma once

#include "batchsensors.h"
#include "network.h"
#include <pthread.h>

#ifndef DEVICE_CODE
#define __host__
#define __device__
#endif

#define __dh_util static inline __device__ __host__


namespace NEAT {

    struct CudaBatchSensorsDims {
        node_size_t nsensors;
        size_t nsteps;
    };

    struct StepParms {
    private:
        __dh_util size_t sizeof_(const CudaBatchSensorsDims &dims) {
            return (sizeof(StepParms) + dims.nsensors * sizeof(real_t));
        }

    public:
        union {
            bool clear_noninput;
            real_t __padding;
        };
        float activations[];

        __dh_util size_t sizeof_buf(const CudaBatchSensorsDims &dims) {
            return dims.nsteps * sizeof_(dims);
        }

        __dh_util StepParms *get(uchar *buf, const CudaBatchSensorsDims &dims, size_t istep) {
            return (StepParms *)(buf + sizeof_(dims)*istep);
        }
    };

    struct CudaLink {
        link_size_t partition;
        node_size_t in_node_index;
        real_t weight;
    };

    struct ActivationPartition {
        node_size_t out_node_index;
        uchar offset;
        uchar len;
    };

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

    struct Lens {
        uint main;
        uint input;
        uint output;
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

    class CudaBatchSensors : public BatchSensors {
    private:
        CudaBatchSensorsDims dims;
        uchar *h_buf;

    public:
        CudaBatchSensors(const CudaBatchSensorsDims &dims);
        virtual ~CudaBatchSensors();

        uint sizeof_buf() {return StepParms::sizeof_buf(dims);}
        uchar *get_h_buf() {return h_buf;};
        const CudaBatchSensorsDims &get_dims() {return dims;}

        virtual void configure_step(size_t istep,
                                    const std::vector<real_t> &values,
                                    bool clear_noninput);

    };

    class CudaNetwork : public Network {
    private:
        friend class CudaNetworkBatch;

        CudaNetDims dims;
        std::vector<CudaLink> gpu_links;
        std::vector<ActivationPartition> partitions;

        RawBuffers bufs;
        Offsets offsets;
        size_t const *istep_output;

    public:
        CudaNetwork() {}
        virtual ~CudaNetwork() {}

        void configure_batch(const RawBuffers &bufs,
                             const Offsets &offsets,
                             size_t const *istep_output);

        virtual void configure(const NetDims &counts,
                               NetNode *nodes,
                               NetLink *links);

        virtual NetDims get_dims() { return dims; }

        virtual real_t get_output(size_t index);
    };

    class CudaNetworkBatch {
        int device;
        uint nnets;
        size_t istep_output;
        GpuState *h_gpu_states;
        GpuState *d_gpu_states;

        CudaBatchSensorsDims sensor_dims;
        RawBuffers h_bufs;
        RawBuffers d_bufs;
        Offsets offsets;
        Lens capacity;
        Lens lens;
        uint sizeof_shared;

    public:
        CudaNetworkBatch(int device, uint nnets);
        ~CudaNetworkBatch();

        void configure(CudaBatchSensors *batch_sensors,
                       CudaNetwork **nets,
                       uint nnets);

        void activate(uint ncycles);

        void set_output_step(size_t istep);
    };

    void test_sum_partition();
}
