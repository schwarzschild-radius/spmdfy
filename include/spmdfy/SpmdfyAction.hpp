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
#include <memory>
#include <string>
#include <vector>

// third party headers
#include <nlohmann/json.hpp>

// spmdfy headers
#include <spmdfy/Generator/SimpleGenerator.hpp>
#include <spmdfy/Generator/CFGGenerator/CFGGenerator.hpp>
#include <spmdfy/SpmdfyStmtVisitor.hpp>
#include <spmdfy/utils.hpp>

namespace nl = nlohmann;
namespace ct = clang::tooling;
namespace mat = clang::ast_matchers;

namespace spmdfy {

class SpmdfyConsumer : public clang::ASTConsumer {
  public:
    explicit SpmdfyConsumer(clang::ASTContext *m_context,
                            std::ostringstream &file_writer)
        : m_context(*m_context), m_sm(m_context->getSourceManager()) {
        this->m_lang_opts = m_context->getLangOpts();
        this->gen = llvm::make_unique<CFGGenerator>(*m_context, file_writer);
    }
    virtual void HandleTranslationUnit(clang::ASTContext &m_context);

  private:
    std::unique_ptr<ISPCGenerator> gen;
    clang::ASTContext &m_context;
    clang::SourceManager &m_sm;
    clang::LangOptions m_lang_opts;
};

class SpmdfyAction : public clang::ASTFrontendAction {

  public:
    SpmdfyAction(std::ostringstream &file_writer)
        : m_file_writer(file_writer) {}
    virtual auto CreateASTConsumer(clang::CompilerInstance &Compiler,
                                   llvm::StringRef InFile)
        -> std::unique_ptr<clang::ASTConsumer> override;

  private:
    std::ostringstream& m_file_writer;
};

class SpmdfyFrontendActionFactory : public clang::tooling::FrontendActionFactory{
    public:
        template<typename ...ParamsTy>
        SpmdfyFrontendActionFactory(ParamsTy&... params) : action(new SpmdfyAction(params...)) {}
        virtual auto create() -> clang::FrontendAction * override;
    private:
        clang::FrontendAction* action;
};

} // namespace spmdfy

#endif