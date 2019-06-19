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
} // namespace spmdfy