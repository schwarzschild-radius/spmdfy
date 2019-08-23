#include <map>
#include <string>

/// A Type map between CUDA/C++ types to ISPC types
extern const std::map<std::string, std::string> g_SpmdfyTypeMap;

/// A Function map between CUDA atomics to ISPC atomics
extern const std::map<std::string, std::string> g_SpmdfyAtomicMap;

/// A Function map between CUDA MathIntrinsics to ISPC Math Functions
extern const std::map<std::string, std::string> g_SpmdfyMathInstrinsicsMap;