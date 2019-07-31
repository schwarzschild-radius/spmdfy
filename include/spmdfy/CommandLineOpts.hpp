#include <llvm/Support/CommandLine.h>

extern llvm::cl::OptionCategory spmdfy_options;
extern llvm::cl::opt<std::string> output_filename;
extern llvm::cl::opt<bool> dump_json;
extern llvm::cl::opt<bool> verbosity;