#include "reduce.cuh"

__global__ void reduce(int *a, int *partial_sum, int N) {
    size_t tid = threadIdx.x;
    size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    for (size_t s = N / 2; s > 0; s >>= 1) {
        if (tid < s) {
            a[gid] += a[gid + s];
        }
        __syncthreads();
    }
    if (tid == 0)
        partial_sum[blockIdx.x] = a[blockIdx.x * blockDim.x];
}