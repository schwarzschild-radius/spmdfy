#ifndef PASS_MANAGER_HPP
#define PASS_MANAGER_HPP

#include <functional>
#include <tuple>
#include <vector>

#include <spmdfy/Generator/CFGGenerator/CFG.hpp>
#include <spmdfy/utils.hpp>

#include <spmdfy/Pass/Passes/PrintCFGPass.hpp>

#include <spmdfy/Pass/PassSequence.hpp>

#include <clang/AST/RecursiveASTVisitor.h>

namespace spmdfy {

namespace pass {

class PassManager {
  public:
    using SpmdTUTy = std::vector<CFG::CFGNode *>;

    PassManager(clang::ASTContext &ast_context, SpmdTUTy &spmd_tutbl)
        : m_spmd_tutbl(spmd_tutbl), m_ast_context(ast_context),
          m_sm(ast_context.getSourceManager()),
          m_lang_opts(ast_context.getLangOpts()) {
        m_lang_opts.CPlusPlus = true;
        m_lang_opts.Bool = true;
        initPassSequence();
    }

    auto initPassSequence() -> void;

    auto runPassSequence() -> bool;

  private:
    // AST Specfic variables
    clang::ASTContext &m_ast_context;
    clang::SourceManager &m_sm;
    clang::LangOptions m_lang_opts;

    // CFG specific variables
    SpmdTUTy &m_spmd_tutbl;
    pass_sequence_t pass_sequence;
};

} // namespace pass

} // namespace spmdfy
#endif