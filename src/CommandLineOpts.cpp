#include <spmdfy/CommandLineOpts.hpp>

llvm::cl::OptionCategory spmdfy_options("spmdfy");

llvm::cl::opt<std::string>
    output_filename("o", llvm::cl::desc("Specify Ouput Filename"),
                    llvm::cl::value_desc("filename"),
                    llvm::cl::cat(spmdfy_options));

llvm::cl::opt<bool>
    verbosity("v",
              llvm::cl::desc("Show commands to run and use verbose output"),
              llvm::cl::cat(spmdfy_options));

llvm::cl::opt<bool> toggle_ispc_macros(
    "fno-ispc-macros",
    llvm::cl::desc(
        "toggle generation of ispc macros in the current output file"),
    llvm::cl::cat(spmdfy_options));

llvm::cl::opt<std::string> generate_ispc_macros(
    "generate-ispc-macros", llvm::cl::desc("File containing ISPC Macros"),
    llvm::cl::value_desc("filename"), llvm::cl::cat(spmdfy_options));