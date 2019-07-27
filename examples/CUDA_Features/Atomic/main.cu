#include <iostream>
#include <vector>

#include "atomic.cuh"
#include "atomic.h"
#include "cuda_utils.cuh"

int log2(int i) {
    int r = 0;
    while (i >>= 1)
        r++;
    return r;
}

int bit_reverse(int w, int bits) {
    int r = 0;
    for (int i = 0; i < bits; i++) {
        int bit = (w & (1 << i)) >> i;
        r |= bit << (bits - i - 1);
    }
    return r;
}

template <typename T>
void executeCUDA(size_t ARRAY_SIZE, size_t BIN_COUNT, const T *h_in,
                 const T *h_bins, T *cuda_bins) {
    T *d_in;
    T *d_bins;

    size_t BIN_BYTES = BIN_COUNT * sizeof(T);
    size_t ARRAY_BYTES = ARRAY_SIZE * sizeof(T);

    cudaMalloc((void **)&d_in, ARRAY_BYTES);
    cudaMalloc((void **)&d_bins, BIN_BYTES);

    cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bins, h_bins, BIN_BYTES, cudaMemcpyHostToDevice);

    atomic<<<ARRAY_SIZE / 64, 64>>>(d_bins, d_in, BIN_COUNT);

    cudaMemcpy(cuda_bins, d_bins, BIN_BYTES, cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_bins);
}

template <typename T>
void executeISPC(size_t ARRAY_SIZE, size_t BIN_COUNT, const T *h_in,
                 const int *h_bins, int *ispc_bins) {
    ispc::Dim3 grid_dim{static_cast<int32_t>(ARRAY_SIZE / 64), 1, 1};
    ispc::Dim3 block_dim{64, 1, 1};
    ispc::atomic(grid_dim, block_dim, 0, ispc_bins, h_in, BIN_COUNT);
}

template <typename T>
void executeReference(size_t ARRAY_SIZE, size_t BIN_COUNT, const T *h_in,
                 const int *h_bins, int *ref_bins) {
    for(int i = 0; i < ARRAY_SIZE; i++){
        int my_item = h_in[i];
        int my_bin = my_item % BIN_COUNT;
        ref_bins[my_bin] += 1;
    }
}

int main(int argc, char **argv) {
    const int ARRAY_SIZE = 65536;
    const int BIN_COUNT = 16;

    // generate the input array on the host
    std::vector<int> h_in(ARRAY_SIZE), h_bins(BIN_COUNT), ref_bins(BIN_COUNT), cuda_bins(BIN_COUNT), ispc_bins(BIN_COUNT);
    for (int i = 0; i < ARRAY_SIZE; i++) {
        h_in[i] = bit_reverse(i, log2(ARRAY_SIZE));
    }
    for (int i = 0; i < BIN_COUNT; i++) {
        h_bins[i] = 0;
        ispc_bins[i] = 0;
    }
    executeReference(ARRAY_SIZE, BIN_COUNT, h_in.data(), h_bins.data(), ref_bins.data());
    executeCUDA(ARRAY_SIZE, BIN_COUNT, h_in.data(), h_bins.data(), cuda_bins.data());
    executeISPC(ARRAY_SIZE, BIN_COUNT, h_in.data(), h_bins.data(), ispc_bins.data());
    if(checkResults(BIN_COUNT, ref_bins, cuda_bins, ispc_bins))
        return 1;

    return 0;
}