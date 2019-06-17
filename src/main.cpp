// clang headers
#include <clang/Tooling/CommonOptionsParser.h>
#include <clang/Tooling/Tooling.h>

// llvm headers
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/Path.h>

// spmdfy headers
#include <fstream>
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
    std::ostringstream function_string;
    function_string << "export ";
    function_string << "void " << name << R"(
        (uniform Dim3& gridDim, uniform Dim3& blockDim
    )";

    for (std::string param : metadata["params"]) {
        function_string << ", uniform " << param << '\n';
    }

    function_string << "){\n";


    function_string << "}\n";
    return function_string.str();
}

std::string generateISPCFunction(std::string name, nl::json metadata) {
    std::ostringstream function_string;
    function_string << metadata["return_type"].c_str() << " " << name << "(";

    function_string << "}\n";
    return function_string.str();
}

std::string getFilenameFromSource(std::string filepath) {
    const auto [_, filename] = llvm::StringRef(filepath).split('/');
    return "";
}

llvm::cl::OptionCategory spmdfy_options("spmdfy -help");
llvm::cl::opt<std::string>
    output_filename("o", llvm::cl::desc("Specify Ouput Filename"),
                    llvm::cl::cat(spmdfy_options));

int main(int argc, const char **argv) {
    using namespace clang::tooling;
    CommonOptionsParser options_parser(argc, argv, spmdfy_options,
                                       llvm::cl::Optional);
    ClangTool tool(options_parser.getCompilations(),
                   options_parser.getSourcePathList());
    std::vector<std::string> file_sources = options_parser.getSourcePathList();
    std::error_code error_code;
    std::string filename = output_filename == ""
                               ? ("./" + getFilenameFromSource(file_sources[0]))
                               : output_filename;
    llvm::errs() << "Writing to : " << filename << '\n';
    std::fstream out_file(filename, std::ios_base::out);

    for (auto &src : file_sources) {
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
        tool.appendArgumentsAdjuster(getInsertArgumentAdjuster(
            "./include", ArgumentInsertPosition::BEGIN));
        tool.appendArgumentsAdjuster(getInsertArgumentAdjuster(
            "-isystem", ArgumentInsertPosition::BEGIN));
        tool.appendArgumentsAdjuster(getInsertArgumentAdjuster(
            "./include/cuda_wrappers", ArgumentInsertPosition::BEGIN));
        tool.appendArgumentsAdjuster(getInsertArgumentAdjuster(
            "-isystem", ArgumentInsertPosition::BEGIN));
        tool.appendArgumentsAdjuster(getInsertArgumentAdjuster(
            "-std=c++17", ArgumentInsertPosition::BEGIN));
        spmdfy::SpmdfyAction action;
        tool.run(newFrontendActionFactory(&action).get());
        nl::json metadata = action.getMetadata();

        std::string ispc_dim_struct = "struct Dim3{\n"
                                      "     uniform int x, y, z;\n"
                                      "};\n";

        out_file << ispc_dim_struct;

        for (auto &[name, data] : metadata.items()) {
            if(data["exported"])
                out_file << generateISPCKernel(name, data) << '\n';
            else
                out_file << generateISPCFunction(name, data) << '\n';
        }
    }
    out_file.close();
    return 0;
}