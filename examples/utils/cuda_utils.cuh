#define CUDACheck(stmt)                                                        \
    {                                                                          \
        cudaError_t err = stmt;                                                \
        if (err != cudaSuccess) {                                              \
            std::cerr << "Failed to run " << #stmt << std::endl;               \
            std::cerr << cudaGetErrorString(err) << std::endl;                 \
        }                                                                      \
    }