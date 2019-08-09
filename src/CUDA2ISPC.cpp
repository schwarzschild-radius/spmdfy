#include <spmdfy/CUDA2ISPC.hpp>

const std::map<std::string, std::string> g_SpmdfyTypeMap = {
    {"char", "int8"},     {"unsigned char", "unsigned int8"},
    {"int32_t", "int32"}, {"unsigned int32_t", "unsigned int32"},
    {"int64_t", "int64"}, {"unsigned int64_t", "unsigned int64"},
    {"short", "int16"},   {"unsigned short", "unsigned int16"},
    {"_Bool", "bool"}};

const std::map<std::string, std::string> g_SpmdfyAtomicMap = {
    {"atomicAdd", "atomic_add_global"},
    {"atomicSub", "atomic_subtract_global"},
    {"atomicExch", "atomic_swap_global"},
    {"atomicMin", "atomic_min_global"},
    {"atomicMax", "atomic_max_global"},
    {"atomicCAS", "atomic_compare_exchange"}};

const std::map<std::string, std::string> g_SpmdfyMathInstrinsicsMap;

const std::string ISPCMacros = R"(
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
            ISPC_GRID                                                                  \
            ISPC_BLOCK

        #define ISPC_END                                                               \
            ISPC_GRID_END                                                              \
            ISPC_BLOCK_END

        #define SYNCTHREADS()                                                          \
            ISPC_BLOCK_END                                                             \
            ISPC_BLOCK

        // CUDA dim3 struct
        struct Dim3{
            unsigned int32 x, y, z;
        };
    )";