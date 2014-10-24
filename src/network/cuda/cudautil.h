#pragma once

#include "neattypes.h"

using namespace NEAT;

#define xcuda(stmt) {                                                   \
        cudaError_t err = stmt;                                         \
        if (err != cudaSuccess) {                                       \
            std::cerr << __FILE__ << ":" << __LINE__ << ": Failed to run " << #stmt << ". Reason: " << cudaGetErrorString(err) << std::endl; \
            abort();                                                    \
        }                                                               \
    }

#define p(msg) std::cout << "[cuda]: " << msg << std::endl

#define errif( STMT, MSG... ) if( STMT ) { fprintf(stderr, "[%s:%d] '%s' ", __FILE__, __LINE__, #STMT); fprintf(stderr, MSG); fprintf(stderr, "\n"); abort(); }

#define require( STMT ) if( !(STMT) ) { fprintf(stderr, "ASSERTION ERROR! [%s:%d] '%s'\n", __FILE__, __LINE__, #STMT); abort(); }

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

    template<typename T>
    static void free_host(__inout T *&buf, bool tolerate_shutdown = false) {
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

    template<typename T>
    static void free_dev(T *&buf) {
        if(buf) {
            xcuda( cudaFree(buf) );
            buf = 0;
        }
    }

    template<typename T>
    static void grow_buffers(__inout T *&h_buf, __inout T *&d_buf,
                             __inout uint &capacity, __in uint newlen) {
        if(newlen > capacity) {
            free_host(h_buf);
            free_dev(d_buf);
            capacity = newlen;
            h_buf = alloc_host(newlen);
            d_buf = alloc_dev(newlen);
        }
    }

