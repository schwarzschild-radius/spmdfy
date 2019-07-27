#include <iostream>
#include <vector>

#define CUDACheck(stmt)                                                        \
    {                                                                          \
        cudaError_t err = stmt;                                                \
        if (err != cudaSuccess) {                                              \
            std::cerr << "Failed to run " << #stmt << std::endl;               \
            std::cerr << cudaGetErrorString(err) << std::endl;                 \
        }                                                                      \
    }

bool checkResults(size_t N, std::vector<int> ref, std::vector<int> cuda, std::vector<int> ispc) {
    bool result = false;
    for(size_t i = 0; i < N; i++){
        if(ref[i] != cuda[i] || ref[i] != ispc[i]){
            std::cerr << "Mismatch at index : " << i << " " << cuda[i] << ", " << ispc[i] << "\n";
            result = true;
        }
    }
    return result;
}