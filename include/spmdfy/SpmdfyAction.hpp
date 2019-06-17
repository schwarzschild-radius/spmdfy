// libtooling headers
#include <clang/AST/ASTConsumer.h>
#include <clang/AST/RecursiveASTVisitor.h>
#include <clang/AST/StmtVisitor.h>
#include <clang/ASTMatchers/ASTMatchFinder.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/FrontendAction.h>
#include <clang/Tooling/Tooling.h>

// llvm headers
#include <llvm/Support/Debug.h>

// standard headers
#include <algorithm>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

// third party headers
#include <nlohmann/json.hpp>

namespace nl = nlohmann;
namespace ct = clang::tooling;
namespace mat = clang::ast_matchers;

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

// works only on the body
class SpmdfyStmtVisitor
    : public clang::StmtVisitor<SpmdfyStmtVisitor, clang::Stmt *> {
  public:
    SpmdfyStmtVisitor(clang::SourceManager &sm)
        : m_sm(sm) {}
    clang::Stmt *VisitCompoundStmt(clang::CompoundStmt *C);

  private:
    clang::SourceManager &m_sm;
    clang::LangOptions m_lang_opt;
};

class SpmdfyAction : public clang::ASTFrontendAction,
                     public mat::MatchFinder::MatchCallback {

  public:
    explicit SpmdfyAction()
        : clang::ASTFrontendAction() {}
    bool cudaKernelFunction(const mat::MatchFinder::MatchResult &result);
    bool cudaDeviceFunction(const mat::MatchFinder::MatchResult &result);
    std::unique_ptr<clang::ASTConsumer> newASTConsumer();
    std::unique_ptr<clang::ASTConsumer>
    CreateASTConsumer(clang::CompilerInstance &, llvm::StringRef) override;
    nl::json getMetadata() { return m_function_metadata; }

  protected:
    void run(const mat::MatchFinder::MatchResult &result) override;

  private:
    std::unique_ptr<mat::MatchFinder> m_finder;
    nl::json m_function_metadata;
    SpmdfyStmtVisitor* stmt_visitor;
};

} // namespace spmdfy