#include <spmdfy/CUDA2ISPC.hpp>

const std::map<std::string, std::string> g_SpmdfyTypeMap = {
    {"char", "int8"},
    {"unsigned char", "unsigned int8"},
    {"int32_t", "int32"},
    {"uint32_t", "int32"},
    {"unsigned int32_t", "unsigned int32"},
    {"uint32_t", "unsigned int32"},
    {"int8_t", "int8"},
    {"uint8_t", "unsigned int8"},
    {"uint", "unsigned int"},
    {"int64_t", "int64"},
    {"long", "int64"},
    {"unsigned long", "unsigned int64"},
    {"uint64_t", "unsigned int64"},
    {"short", "int16"},
    {"unsigned short", "unsigned int16"},
    {"_Bool", "bool"}};

const std::map<std::string, std::string> g_SpmdfyAtomicMap = {
    {"atomicAdd", "atomic_add_global"},
    {"atomicSub", "atomic_subtract_global"},
    {"atomicExch", "atomic_swap_global"},
    {"atomicMin", "atomic_min_global"},
    {"atomicMax", "atomic_max_global"},
    {"atomicCAS", "atomic_compare_exchange"}};

const std::map<std::string, std::pair<bool, std::string>>
    g_SpmdfyMathInstrinsicsMap = {{"aconf", {1, "acosf"}},
                                  {"acoshf", {0, R"begin(
    )begin"}},
                                  {"asinf", {1, "asin"}},
                                  {"asinhf", {0, R"begin(
        float asinhf(float x){
            return (exp(x) - exp(-x)) / 2;
        }
    )begin"}},
                                  {"atan2f", {1, "atan2"}},
                                  {"atanf", {1, "atan"}},
                                  {"atanhf", {0, R"begin(
        float atanhf(float x){
            return ((exp(x) - exp(-x)) / (exp(x) + exp(-x)));
        }
    )begin"}},
                                  {"cbrtf", {0, R"begin()begin"}},
                                  {"ceilf", {1, "ceil"}},
                                  {"copysignf", {0, R"begin(
        float copysignf(float x, float y){
            return x * (y < 0 ? -1 : 1);
        }
    )begin"}},
                                  {"cosf", {1, "cos"}},
                                  {"coshf", {0, R"begin(
        float coshf(float x){
            return (exp(x) + exp(-x)) / 2;
        }
    )begin"}},
                                  {"__expf", {1, "exp"}},
                                  {"__logf", {1, "log"}},
                                  {"sqrtf", {1, "sqrt"}},
                                  {"__ffs", {0, R"begin(
        int __ffs(int i) {
            return ((i != 0) ? count_trailing_zeros(i) + 1 : 0);
        }
                                  )begin"}}};