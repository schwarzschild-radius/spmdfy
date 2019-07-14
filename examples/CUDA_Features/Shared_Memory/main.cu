#include <iostream>
#include <cuda_runtime.h>
#include "shared_memory.cuh"
#include "shared_memory_ispc.h"
#include "cuda_utils.cuh"

void executeCUDA(int *a, int *r, int *d, int n) {
    int *d_d;
    cudaMalloc(&d_d, n * sizeof(int));
    cudaMemcpy(d_d, a, n * sizeof(int), cudaMemcpyHostToDevice);
    staticReverse<<<1, n>>>(d_d, n);
    cudaMemcpy(d, d_d, n * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++)
        if (d[i] != r[i])
            printf("Error: d[%d]!=r[%d] (%d, %d)\n", i, i, d[i], r[i]);

    cudaMemcpy(d_d, a, n * sizeof(int), cudaMemcpyHostToDevice);
    dynamicReverse<<<1, n, n * sizeof(int)>>>(d_d, n);
    cudaMemcpy(d, d_d, n * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++)
        if (d[i] != r[i])
            printf("Error: d[%d]!=r[%d] (%d, %d)\n", i, i, d[i], r[i]);
}

void executeISPC(int *a, int *r, int *d, int n) {
    int* d_d = (int*)malloc(n * sizeof(int));
    memcpy(d_d, a, n * sizeof(int));
    ispc::staticReverse({1, 1, 1}, {n, 1, 1}, 0, d_d, n);
    memcpy(d, d_d, n * sizeof(int));
    for (int i = 0; i < n; i++)
        if (d[i] != r[i])
            printf("Error: d[%d]!=r[%d] (%d, %d)\n", i, i, d[i], r[i]);
    memcpy(d_d, a, n * sizeof(int));            
    ispc::dynamicReverse({1, 1, 1}, {n, 1, 1}, n * sizeof(int), d_d, n);
    memcpy(d, d_d, n * sizeof(int));
    for (int i = 0; i < n; i++)
        if (d[i] != r[i])
            printf("Error: d[%d]!=r[%d] (%d, %d)\n", i, i, d[i], r[i]);
}

template <typename Ref, typename... T>
void compareResults(int N, Ref &ref, T &... rest) {
    for (int i = 0; i < N; i++) {
        if (((ref[i] != rest[i]) || ...)) {
            std::cerr << "error at " << i << " " << ref[i] << " ";
            (std::cerr << ... << rest[i]) << '\n';
        }
    }
}

int main(void) {
    const int n = 64;
    int a[n], r[n], d[n];

    for (int i = 0; i < n; i++) {
        a[i] = i;
        r[i] = n - i - 1;
        d[i] = 0;
    }

    executeCUDA(a, r, d, n);
    executeISPC(a, r, d, n);
    return 0;
}