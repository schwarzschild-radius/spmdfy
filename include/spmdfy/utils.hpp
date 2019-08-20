#ifndef SPMDFY_UTILS_HPP
#define SPMDFY_UTILS_HPP

// clang headers
#include <clang/Basic/LangOptions.h>
#include <clang/Basic/SourceManager.h>
#include <clang/Lex/Lexer.h>

// llvm headers
#include <llvm/Support/Path.h>

// standard header
#include <type_traits>
#include <variant>

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

template <typename NodeTy>
static bool isExpansionInMainFile(clang::SourceManager &sm, NodeTy *node) {
    return sm.isInMainFile(sm.getExpansionLoc(node->getBeginLoc()));
}

template <typename Iter>
std::string strJoin(Iter b, Iter e, std::string sep = ", ") {
    std::string join = *b;
    while (++b != e) {
        join += (sep + *b);
    }
    return join;
}

template <typename... T> struct Overload : T... {
    using T::operator()...;
    Overload(T &&... ts) : T(std::forward<std::decay_t<T>>(ts))... {}
};

std::string getFileNameFromSource(std::string filepath);
std::string getAbsoluteFilePath(const std::string &sFile, std::error_code &EC);

#define SRCDUMP(NODE) sourceDump(m_sm, m_lang_opts, NODE)
} // namespace spmdfy

#endif