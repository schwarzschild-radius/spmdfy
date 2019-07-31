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