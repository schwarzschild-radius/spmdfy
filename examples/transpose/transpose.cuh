#include <cuda_runtime.h>

__global__ void transpose_parallel_per_element(int a[], int b[], size_t N,
                                               size_t K);