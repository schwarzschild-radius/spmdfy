/** \file SpmdfyAction.hpp
 *  \brief Classes for accessing AST through libTooling and drive the ISPC
 * generation
 *
 *  \author Pradeep Kumar  (schwarzschild-radius/@pt_of_no_return)
 *  \bug No know bugs
 *  \defgroup Frontend
 * */

#ifndef SPMDFY_ACTION_HPP
#define SPMDFY_ACTION_HPP

// libtooling headers
#include <clang/AST/ASTConsumer.h>
#include <clang/AST/RecursiveASTVisitor.h>
#include <clang/ASTMatchers/ASTMatchFinder.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/FrontendAction.h>
#include <clang/Lex/PPCallbacks.h>
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
#include <spmdfy/Generator/CFGGenerator/CFGGenerator.hpp>
#include <spmdfy/Generator/SimpleGenerator.hpp>
#include <spmdfy/utils.hpp>

namespace nl = nlohmann;
namespace ct = clang::tooling;
namespace mat = clang::ast_matchers;

namespace spmdfy {

/**
 * \class SpmdfyConsumer
 * \ingroup Frontend
 *
 * \brief Class initiates the translation phase by providing the handle to AST
 * to the ISPCGenerator
 *
 * */
class SpmdfyConsumer : public clang::ASTConsumer {
  public:
    explicit SpmdfyConsumer(clang::ASTContext *m_context,
                            std::ostringstream &file_writer)
        : m_context(*m_context), m_sm(m_context->getSourceManager()) {
        this->m_lang_opts = m_context->getLangOpts();
        this->gen = llvm::make_unique<CFGGenerator>(*m_context, file_writer);
    }

    /// creates the handle translation unit and passes it to the codegen methods
    virtual void HandleTranslationUnit(clang::ASTContext &m_context);

  private:
    std::unique_ptr<ISPCGenerator> gen;
    clang::ASTContext &m_context;
    clang::SourceManager &m_sm;
    clang::LangOptions m_lang_opts;
};

/**
 * \class SpmdfyAction
 * \ingroup Frontend
 *
 * \brief Creates clang tooling drives this class to create a consumer and
 * provides a compiler instance to the consumer. This provides the access the
 * AST for traversal
 *
 * */
class SpmdfyAction : public clang::ASTFrontendAction {

  public:
    SpmdfyAction(std::ostringstream &file_writer)
        : m_file_writer(file_writer) {}
    virtual auto CreateASTConsumer(clang::CompilerInstance &Compiler,
                                   llvm::StringRef InFile)
        -> std::unique_ptr<clang::ASTConsumer> override;

    auto ExecuteAction() -> void override;

    auto InclusionDirective(clang::SourceLocation hash_loc,
                            const clang::Token &include_token,
                            llvm::StringRef file_name, bool is_angled,
                            clang::CharSourceRange filename_range,
                            const clang::FileEntry *file, llvm::StringRef search_path,
                            llvm::StringRef relative_path,
                            const clang::Module *imported,
                            clang::SrcMgr::CharacteristicKind FileType) -> void;

  private:
    std::ostringstream &m_file_writer;
};

/**
 * \class SpmdfyFrontendActionFactory
 * \ingroup Frontend
 *
 * \brief A factory method to create a Frontend action
 *
 * */
class SpmdfyFrontendActionFactory
    : public clang::tooling::FrontendActionFactory {
  public:
    template <typename... ParamsTy>
    SpmdfyFrontendActionFactory(ParamsTy &... params)
        : action(new SpmdfyAction(params...)) {}

    /// creates the frontend action
    virtual auto create() -> clang::FrontendAction * override;

  private:
    clang::FrontendAction *action;
};

} // namespace spmdfy

#endif