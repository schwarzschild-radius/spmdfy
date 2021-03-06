#include <llvm/Support/CommandLine.h>

extern llvm::cl::OptionCategory spmdfy_options;
extern llvm::cl::opt<std::string> output_filename;
extern llvm::cl::opt<bool> verbosity;
extern llvm::cl::opt<bool> toggle_ispc_macros;
extern llvm::cl::opt<std::string> generate_ispc_macros;
extern llvm::cl::opt<bool> generate_decls;