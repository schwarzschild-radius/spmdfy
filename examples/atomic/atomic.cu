#include <cuda_runtime.h>

__global__ void atomic_cuda(int *d_bins, const int *d_in, const int BIN_COUNT) {
    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    int myItem = d_in[myId];
    int myBin = myItem % BIN_COUNT;
    atomicAdd(&(d_bins[myBin]), 1);
}