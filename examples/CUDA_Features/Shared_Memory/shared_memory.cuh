#include <cuda_runtime.h>

__global__ void staticReverse(int *d, int n);

__global__ void dynamicReverse(int *d, int n);