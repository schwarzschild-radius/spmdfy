#ifndef SPMDFY_ACTION_HPP
#define SPMDFY_ACTION_HPP

// libtooling headers
#include <clang/AST/ASTConsumer.h>
#include <clang/AST/RecursiveASTVisitor.h>
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

// spmdfy headers
#include <spmdfy/SpmdfyStmtVisitor.hpp>

namespace nl = nlohmann;
namespace ct = clang::tooling;
namespace mat = clang::ast_matchers;

namespace spmdfy {

class SpmdfyAction : public clang::ASTFrontendAction,
                     public mat::MatchFinder::MatchCallback {

  public:
    explicit SpmdfyAction() : clang::ASTFrontendAction() {}
    // matcher functions
    bool cudaKernelFunction(const mat::MatchFinder::MatchResult &result);
    bool cudaDeviceFunction(const mat::MatchFinder::MatchResult &result);
    bool globalDeclarations(const mat::MatchFinder::MatchResult &result);

    std::unique_ptr<clang::ASTConsumer> newASTConsumer();
    std::unique_ptr<clang::ASTConsumer>
    CreateASTConsumer(clang::CompilerInstance &, llvm::StringRef) override;
    nl::json getMetadata() { return m_function_metadata; }

  protected:
    void run(const mat::MatchFinder::MatchResult &result) override;

  private:
    std::unique_ptr<mat::MatchFinder> m_finder;
    nl::json m_function_metadata;
    SpmdfyStmtVisitor *stmt_visitor;
};

} // namespace spmdfy

#endif