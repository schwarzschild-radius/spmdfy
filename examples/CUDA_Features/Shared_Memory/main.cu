#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#include "cuda_utils.cuh"
#include "shared_memory.cuh"
#include "shared_memory.h"

void executeCUDA(std::vector<int> &a, std::vector<int> &r, std::vector<int> &d,
                 int n) {
    int *d_d = nullptr;
    cudaMalloc(&d_d, n * sizeof(int));
    cudaMemcpy(d_d, a.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    staticReverse<<<1, n>>>(d_d, n);
    cudaMemcpy(d.data(), d_d, n * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++)
        if (d[i] != r[i])
            printf("Error: d[%d]!=r[%d] (%d, %d)\n", i, i, d[i], r[i]);

    cudaMemcpy(d_d, a.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    dynamicReverse<<<1, n, n * sizeof(int)>>>(d_d, n);
    cudaMemcpy(d.data(), d_d, n * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++)
        if (d[i] != r[i])
            printf("Error: d[%d]!=r[%d] (%d, %d)\n", i, i, d[i], r[i]);
}

void executeISPC(std::vector<int> &a, std::vector<int> &r, std::vector<int> &d,
                 int n) {
    int *d_d = (int *)malloc(n * sizeof(int));
    memcpy(d_d, a.data(), n * sizeof(int));
    ispc::staticReverse({1, 1, 1}, {n, 1, 1}, 0, d_d, n);
    memcpy(d.data(), d_d, n * sizeof(int));
    for (int i = 0; i < n; i++)
        if (d[i] != r[i])
            printf("Error: d[%d]!=r[%d] (%d, %d)\n", i, i, d[i], r[i]);
    memcpy(d_d, a.data(), n * sizeof(int));
    ispc::dynamicReverse({1, 1, 1}, {n, 1, 1}, n * sizeof(int), d_d, n);
    memcpy(d.data(), d_d, n * sizeof(int));
    for (int i = 0; i < n; i++)
        if (d[i] != r[i])
            printf("Error: d[%d]!=r[%d] (%d, %d)\n", i, i, d[i], r[i]);
}

int main(void) {
    const int n = 64;
    std::vector<int> a(n), r(n), cuda(n), ispc(n);

    for (int i = 0; i < n; i++) {
        a[i] = i;
        r[i] = n - i - 1;
        ispc[i] = cuda[i] = 0;
    }

    executeCUDA(a, r, cuda, n);
    executeISPC(a, r, ispc, n);

    if (checkResults(n, r, cuda, ispc))
        return 1;
    return 0;
}