#include "saxpy.cuh"

__global__ void saxpy(const int *A, const int *B, int *C, int N, int a) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        C[i] = a * A[i] + B[i];
}