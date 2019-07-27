#include <cuda_runtime.h>

__global__ void saxpy(const int *A, const int *B, int *C, int N, int a);