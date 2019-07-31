#include <spmdfy/CommandLineOpts.hpp>

llvm::cl::OptionCategory spmdfy_options("spmdfy");

llvm::cl::opt<std::string>
    output_filename("o", llvm::cl::desc("Specify Ouput Filename"),
                    llvm::cl::cat(spmdfy_options));

llvm::cl::opt<bool>
    dump_json("dump-json",
              llvm::cl::desc("Jump JSON description of the Translation Unit"),
              llvm::cl::cat(spmdfy_options));
              
llvm::cl::opt<bool>
    verbosity("v",
              llvm::cl::desc("Show commands to run and use verbose output"),
              llvm::cl::value_desc("v"), llvm::cl::cat(spmdfy_options));