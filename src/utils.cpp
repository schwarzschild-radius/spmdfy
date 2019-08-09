#include <spmdfy/utils.hpp>

namespace spmdfy {
std::string sourceDump(const clang::SourceManager &sm,
                       const clang::LangOptions &lang_opt,
                       const clang::SourceLocation &begin,
                       const clang::SourceLocation &end) {
    clang::SourceLocation e(
        clang::Lexer::getLocForEndOfToken(end, 0, sm, lang_opt));
    clang::SourceLocation b(
        clang::Lexer::GetBeginningOfToken(begin, sm, lang_opt));
    if ((sm.getCharacterData(e) - sm.getCharacterData(b)) < 1) {
        llvm::errs() << "Cannot dump source\n";
        return "";
    }
    return std::string(sm.getCharacterData(begin),
                       (sm.getCharacterData(e) - sm.getCharacterData(b)));
}

std::string getFileNameFromSource(std::string filepath) {
    const auto [_, filename] = llvm::StringRef(filepath).rsplit('/');
    return filename;
}

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

} // namespace spmdfy