#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include "saxpy.cuh"
#include "saxpy.h"
#include "cuda_utils.cuh"

void executeCUDA(const std::vector<int> &A, const std::vector<int> &B,
                 std::vector<int> &C, const size_t N, const int a) {
    int *d_A, *d_B, *d_C;
    size_t size_bytes = sizeof(int) * N;

    dim3 blocks = N > 1024 ? (N - 1) / 1024 + 1 : 1,
         threads = N > 1024 ? 1024 : N;

    cudaMalloc((void **)&d_A, size_bytes);
    cudaMalloc((void **)&d_B, size_bytes);
    cudaMalloc((void **)&d_C, size_bytes);

    cudaMemcpy(d_A, A.data(), size_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), size_bytes, cudaMemcpyHostToDevice);

    saxpy<<<blocks, threads>>>(d_A, d_B, d_C, N, a);

    cudaMemcpy(C.data(), d_C, size_bytes, cudaMemcpyDeviceToHost);
}

void executeISPC(const std::vector<int> &A, const std::vector<int> &B,
                 std::vector<int> &C, const size_t N, const int a) {

    ispc::Dim3 blocks{static_cast<int32_t>(N > 1024 ? (N - 1) / 1024 + 1 : 1), 1, 1}, threads{static_cast<int32_t>(N > 1024 ? 1024 : N), 1, 1};
    ispc::saxpy(blocks, threads, 0, A.data(), B.data(), C.data(), N, a);
}

void executeReference(const std::vector<int> &A, const std::vector<int> &B,
                 std::vector<int> &C, const size_t N, const int a){
    for(int i = 0; i < N; i++){
        C[i] = a * A[i] + B[i];
    }
}

int main() {
    size_t N = 100;
    int a = 2;
    std::random_device random_device;
    std::mt19937 random_engine(random_device());
    std::uniform_int_distribution<int> distribution(1, 100);
    std::vector<int> A(N), B(N), ref(N), ispc(N), cuda(N);
    std::generate(A.begin(), A.end(), [&distribution, &random_engine]() -> int {
        return distribution(random_engine);
    });

    std::generate(B.begin(), B.end(), [&distribution, &random_engine]() -> int {
        return distribution(random_engine);
    });

    executeReference(A, B, ref, N, a);
    executeCUDA(A, B, cuda, N, a);
    executeISPC(A, B, ispc, N, a);

    if(compareResults(N, ref, cuda, ispc)){
        return 1;
    }

    return 0;
}