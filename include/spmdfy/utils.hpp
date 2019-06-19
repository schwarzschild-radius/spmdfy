#ifndef SPMDFY_UTILS_HPP
#define SPMDFY_UTILS_HPP

#include <clang/Basic/LangOptions.h>
#include <clang/Basic/SourceManager.h>
#include <clang/Lex/Lexer.h>

namespace spmdfy {
std::string sourceDump(const clang::SourceManager &sm,
                       const clang::LangOptions &lang_opt,
                       const clang::SourceLocation &begin,
                       const clang::SourceLocation &end);

template <typename AstNode>
std::string sourceDump(const clang::SourceManager &sm,
                       const clang::LangOptions &lang_opt, AstNode node) {
    return sourceDump(sm, lang_opt, node->getSourceRange().getBegin(),
                      node->getSourceRange().getEnd());
}

} // namespace spmdfy

#endif