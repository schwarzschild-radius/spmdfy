// clang headers
#include <clang/Tooling/CommonOptionsParser.h>
#include <clang/Tooling/Tooling.h>

// llvm headers
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/Path.h>

// spmdfy headers
#include <spmdfy/Format.hpp>
#include <spmdfy/SpmdfyAction.hpp>

// standard header
#include <fstream>
#include <sstream>

std::string getAbsoluteFilePath(const std::string &sFile, std::error_code &EC) {
    using namespace llvm;
    if (sFile.empty()) {
        return sFile;
    }
    if (!sys::fs::exists(sFile)) {
        llvm::errs() << "\n"
                     << "[SPMDFY] "
                     << "error: "
                     << "source file: " << sFile << " doesn't exist\n";
        EC = std::error_code(
            static_cast<int>(std::errc::no_such_file_or_directory),
            std::generic_category());
        return "";
    }
    SmallString<256> fileAbsPath;
    EC = sys::fs::real_path(sFile, fileAbsPath, true);
    if (EC) {
        llvm::errs() << "\n"
                     << "[SPMDFY] "
                     << "error: " << EC.message() << ": source file: " << sFile
                     << "\n";
        return "";
    }
    EC = std::error_code();
    return fileAbsPath.c_str();
}

std::string generateISPCKernel(std::string name, nl::json metadata) {
    std::string ispc_grid_start = "ISPC_GRID_START\n";
    std::string ispc_block_start = "ISPC_BLOCK_START\n";
    std::string ispc_block_end = "ISPC_BLOCK_END\n";
    std::string ispc_grid_end = "ISPC_GRID_END\n";

    std::ostringstream function_string;
    function_string << "export ";
    function_string << "void " << name << R"(
        (uniform Dim3& gridDim, uniform Dim3& blockDim, 
         uniform unsigned int32 shared_memory_size
        )";

    // shared mem
    for (std::string shmem : metadata["shmem"]) {
        function_string << ", uniform " << shmem << '\n';
    }

    // params
    for (std::string param : metadata["params"]) {
        function_string << ", uniform " << param << '\n';
    }

    function_string << "){\n";

    function_string << "unsigned int<3> blockIdx, threadIdx;\n";
    function_string << ispc_grid_start;

    // body
    for (auto &[block, body] : metadata["body"].items()) {
        function_string << ispc_block_start;
        for (std::string line : body) {
            function_string << line << '\n';
        }
        function_string << ispc_block_end;
    }
    function_string << ispc_grid_end;

    function_string << "}\n";
    return function_string.str();
}

std::string generateISPCFunction(std::string name, nl::json metadata) {
    std::ostringstream function_string;
    function_string << (std::string)metadata["return_type"] << " " << name
                    << "(";
    // params
    for (std::string param : metadata["params"]) {
        function_string << param << '\n';
    }

    function_string << "){\n";

    // body

    function_string << "}\n";
    return function_string.str();
}

std::string generateISPCTranslationUnit(nl::json metadata) {
    std::ostringstream tu;

    // generate macros
    tu << R"(
        #define ISPC_GRID_START                                                        \
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
    )" << '\n';

    // generate Dim3 struct
    tu << R"(
        struct Dim3{
            int x, y, z;
        };
    )";

    for (std::string var_decl : metadata["globals"]) {
        tu << "uniform " << var_decl << '\n';
    }

    for (auto &[name, data] : metadata["function"].items()) {
        if (data["exported"])
            tu << generateISPCKernel(name, data) << '\n';
        else
            tu << generateISPCFunction(name, data) << '\n';
    }
    return tu.str();
}

std::string getFileNameFromSource(std::string filepath) {
    const auto [_, filename] = llvm::StringRef(filepath).rsplit('/');
    return filename;
}

llvm::cl::OptionCategory spmdfy_options("spmdfy -help");
llvm::cl::opt<std::string>
    output_filename("o", llvm::cl::desc("Specify Ouput Filename"),
                    llvm::cl::cat(spmdfy_options));
cl::opt<bool> verbosity("v",
                        cl::desc("Show commands to run and use verbose output"),
                        cl::value_desc("v"), cl::cat(spmdfy_options));

int main(int argc, const char **argv) {
    using namespace clang::tooling;
    CommonOptionsParser options_parser(argc, argv, spmdfy_options,
                                       llvm::cl::Optional);
    ClangTool tool(options_parser.getCompilations(),
                   options_parser.getSourcePathList());
    std::vector<std::string> file_sources = options_parser.getSourcePathList();
    std::error_code error_code;
    std::string filename = (output_filename == "")
                               ? ("./" + getFileNameFromSource(file_sources[0]))
                               : output_filename;
    llvm::errs() << "Writing to : " << filename << '\n';
    std::fstream out_file(filename, std::ios_base::out);

    auto &src = file_sources[0];
    std::string source_abs_path = getAbsoluteFilePath(src, error_code);
    std::string includes =
        "-I" + llvm::sys::path::parent_path(source_abs_path).str();

    tool.appendArgumentsAdjuster(getInsertArgumentAdjuster(
        includes.c_str(), ArgumentInsertPosition::BEGIN));
    tool.appendArgumentsAdjuster(
        getInsertArgumentAdjuster("cuda", ArgumentInsertPosition::BEGIN));
    tool.appendArgumentsAdjuster(
        getInsertArgumentAdjuster("-x", ArgumentInsertPosition::BEGIN));
    tool.appendArgumentsAdjuster(getInsertArgumentAdjuster(
        "--cuda-host-only", ArgumentInsertPosition::BEGIN));
    tool.appendArgumentsAdjuster(
        getInsertArgumentAdjuster("./include", ArgumentInsertPosition::BEGIN));
    tool.appendArgumentsAdjuster(
        getInsertArgumentAdjuster("-isystem", ArgumentInsertPosition::BEGIN));
    tool.appendArgumentsAdjuster(getInsertArgumentAdjuster(
        "./include/cuda_wrappers", ArgumentInsertPosition::BEGIN));
    tool.appendArgumentsAdjuster(
        getInsertArgumentAdjuster("-isystem", ArgumentInsertPosition::BEGIN));
    tool.appendArgumentsAdjuster(
        getInsertArgumentAdjuster("-std=c++17", ArgumentInsertPosition::BEGIN));
    tool.appendArgumentsAdjuster(getClangSyntaxOnlyAdjuster());
    if (verbosity) {
        tool.appendArgumentsAdjuster(
            getInsertArgumentAdjuster("-v", ArgumentInsertPosition::END));
    }

    // run SPMDfy action on the source
    spmdfy::SpmdfyAction action;
    tool.run(newFrontendActionFactory(&action).get());
    nl::json metadata = action.getMetadata();

    out_file << generateISPCTranslationUnit(metadata);
    out_file.close();

    if (spmdfy::format::format(filename))
        llvm::errs() << "Unable to format\n";
    return 0;
}