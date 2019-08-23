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

/**
 * \ingroup Utility
 *
 * \brief Dumps a string from the source file with the location specified by
 * begin and end. It uses the clang's Lexer interface to dump the source
 *
 * */
std::string sourceDump(const clang::SourceManager &sm,
                       const clang::LangOptions &lang_opt,
                       const clang::SourceLocation &begin,
                       const clang::SourceLocation &end);

/// a wrapper method to sourceDump for any ASTNode provided
template <typename AstNode>
std::string sourceDump(const clang::SourceManager &sm,
                       const clang::LangOptions &lang_opt, AstNode node) {
    return sourceDump(sm, lang_opt, node->getSourceRange().getBegin(),
                      node->getSourceRange().getEnd());
}


/**
 * \ingroup Utility
 *
 * \brief Checks if a given ASTNode is in the current translation unit's main section
 *
 * */
template <typename NodeTy>
static bool isExpansionInMainFile(clang::SourceManager &sm, NodeTy *node) {
    return sm.isInMainFile(sm.getExpansionLoc(node->getBeginLoc()));
}

/**
 * \ingroup Utility
 *
 * \brief Joins an array of string by a seperator similar to python's join
 *
 * */
template <typename Iter>
std::string strJoin(Iter b, Iter e, std::string sep = ", ") {
    std::string join = *b;
    while (++b != e) {
        join += (sep + *b);
    }
    return join;
}

/**
 * \class Overload
 * \ingroup Utility
 *
 * \brief Struct that implements the variant overload idiom
 *
 * */
template <typename... T> struct Overload : T... {
    using T::operator()...;
    Overload(T &&... ts) : T(std::forward<std::decay_t<T>>(ts))... {}
};


/// get Filename from Source Path
std::string getFileNameFromSource(std::string filepath);

/// get Absolute File path through llvm's sys module
std::string getAbsoluteFilePath(const std::string &sFile, std::error_code &EC);

/// Macro the the source dump function
#define SRCDUMP(NODE) sourceDump(m_sm, m_lang_opts, NODE)
} // namespace spmdfy

#endif