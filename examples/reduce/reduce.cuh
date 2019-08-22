#include <cstdio>
#include <cuda_runtime.h>

__global__ void reduce(int *a, int *partial_sum, int N);