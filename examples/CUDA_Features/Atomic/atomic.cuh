#include <cuda_runtime.h>

__global__ void atomic(int *d_bins, const int *d_in, const int BIN_COUNT);