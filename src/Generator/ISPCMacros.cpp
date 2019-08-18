#include <string>

namespace spmdfy {

std::string ispc_macros = R"macro(
#define ISPC_GRID_START                                                        \
    Dim3 blockIdx, threadIdx;                                                  \
    for (blockIdx.z = 0; blockIdx.z < gridDim.z; blockIdx.z++) {               \
        for (blockIdx.y = 0; blockIdx.y < gridDim.y; blockIdx.y++) {           \
            for (blockIdx.x = 0; blockIdx.x < gridDim.x; blockIdx.x++) {

#define ISPC_BLOCK_START                                                       \
    for (threadIdx.z = 0; threadIdx.z < blockDim.z; threadIdx.z++) {           \
        for (threadIdx.y = 0; threadIdx.y < blockDim.y; threadIdx.y++) {       \
            for (threadIdx.x = programIndex; threadIdx.x < blockDim.x;         \
                 threadIdx.x += programCount) {

#define ISPC_GRID_END                                                          \
    }                                                                          \
    }                                                                          \
    }

#define ISPC_BLOCK_END                                                         \
    }                                                                          \
    }                                                                          \
    }

#define ISPC_START                                                             \
    ISPC_GRID_START                                                            \
    ISPC_BLOCK_START

#define ISPC_END                                                               \
    ISPC_GRID_END                                                              \
    ISPC_BLOCK_END

#define SYNCTHREADS()                                                          \
    ISPC_BLOCK_END                                                             \
    ISPC_BLOCK_START

#define ISPC_KERNEL(function, ...)                                             \
    export void function(                                                      \
        const uniform Dim3 &gridDim, const uniform Dim3 &blockDim,             \
        const uniform size_t &shared_memory_size, __VA_ARGS__)

#define ISPC_DEVICE_FUNCTION(rety, function, ...)                              \
    rety function(const uniform Dim3 &gridDim, const uniform Dim3 &blockDim,   \
                  const Dim3 &blockIdx, const Dim3 &threadIdx, __VA_ARGS__)

#define ISPC_DEVICE_CALL(function, ...)                                        \
    function(gridDim, blockDim, blockIdx, threadIdx, __VA_ARGS__)

#define NS(x, y) x##_##y
#define NS3(x, y, z) x##_##y##_##z
#define ENUM(x, y) const int x##_##y
struct Dim3 {
    int x, y, z;
};

)macro";

}