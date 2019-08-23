// clang headers
#include <clang/Tooling/CommonOptionsParser.h>
#include <clang/Tooling/Tooling.h>

// spmdfy headers
#include <spmdfy/CommandLineOpts.hpp>
#include <spmdfy/Format.hpp>
#include <spmdfy/Logger.hpp>
#include <spmdfy/SpmdfyAction.hpp>

// standard header
#include <fstream>
#include <sstream>

namespace spmdfy {
extern std::string ispc_macros;
}

int main(int argc, const char **argv) {
    spmdfy::Logger::initLogger();
    using namespace clang::tooling;
    CommonOptionsParser options_parser(argc, argv, spmdfy_options,
                                       llvm::cl::Optional);
    ClangTool tool(options_parser.getCompilations(),
                   options_parser.getSourcePathList());
    std::vector<std::string> file_sources = options_parser.getSourcePathList();

    if (generate_ispc_macros != std::string()) {
        SPMDFY_INFO("Writing ispc macros to : {}", generate_ispc_macros);
        std::fstream out_file(generate_ispc_macros, std::ios_base::out);
        out_file << spmdfy::ispc_macros;
        out_file.close();
        return 0;
    }

    if (file_sources.empty()) {
        llvm::cl::PrintHelpMessage();
        return 1;
    }

    std::error_code error_code;

    std::string &src = file_sources[0];
    std::string source_abs_path = spmdfy::getAbsoluteFilePath(src, error_code);
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

    std::ostringstream tu_stream;

    // run SPMDfy action on the source
    spmdfy::SpmdfyFrontendActionFactory action(tu_stream);
    if (tool.run(&action)) {
        SPMDFY_ERROR("error: unable to spmdfy file");
        return 1;
    }

    if (output_filename != "") {
        SPMDFY_INFO("Writing to : {}", output_filename);
        std::fstream out_file(output_filename, std::ios_base::out);
        if (!toggle_ispc_macros) {
            out_file << spmdfy::ispc_macros;
        } else {
            out_file << "#include \"ISPCMacros.ispc.h\""
                     << "\n";
        }
        out_file << tu_stream.str();
        out_file.close();
        if (spmdfy::format::format(output_filename))
            SPMDFY_ERROR("Unable to format");
    }
    return 0;
}