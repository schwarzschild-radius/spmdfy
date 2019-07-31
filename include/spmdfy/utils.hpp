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

template<typename Iter>
std::string strJoin(Iter b, Iter e, char sep=','){
    std::string temp = *b;
    while(++b != e){
        temp += (std::to_string(sep) + " " + *b);
    }
    return temp;
}
} // namespace spmdfy

#endif