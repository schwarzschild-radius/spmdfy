#ifndef PASS_MANAGER_HPP
#define PASS_MANAGER_HPP

#include <functional>
#include <tuple>
#include <vector>

#include <spmdfy/CFG/CFG.hpp>
#include <spmdfy/utils.hpp>

// clang-format off
#include <spmdfy/Pass/Passes/LocateASTNodes.hpp>
#include <spmdfy/Pass/Passes/InsertISPCNodes.hpp>
#include <spmdfy/Pass/Passes/HoistShmemNodes.hpp>
#include <spmdfy/Pass/Passes/DuplicatePartialNodes.hpp>
#include <spmdfy/Pass/Passes/DetectPartialNodes.hpp>
#include <spmdfy/Pass/Passes/PrintReverseCFGPass.hpp>
#include <spmdfy/Pass/Passes/PrintCFGPass.hpp>
// clang-format on

#include <spmdfy/Pass/PassSequence.hpp>
#include <spmdfy/Pass/PassWorkspace.hpp>

#include <clang/AST/RecursiveASTVisitor.h>

namespace spmdfy {

namespace pass {

/**
 * \class PassManager
 * \ingroup Pass
 *
 * \brief The class that drives the pass sequence on the CFG in the given order
 * in sequence
 *
 * */

class PassManager {
  public:
    using SpmdTUTy = std::vector<cfg::CFGNode *>;

    PassManager(clang::ASTContext &ast_context, SpmdTUTy &spmd_tutbl)
        : m_spmd_tutbl(spmd_tutbl), m_ast_context(ast_context),
          m_sm(ast_context.getSourceManager()),
          m_lang_opts(ast_context.getLangOpts()) {
        m_lang_opts.CPlusPlus = true;
        m_lang_opts.Bool = true;
        initPassSequence();
    }

    /// sets the arguments for the sequence
    auto initPassSequence() -> void;

    /// run the sequence on the CFG
    auto runPassSequence() -> bool;

  private:
    // AST Specfic variables
    clang::ASTContext &m_ast_context;
    clang::SourceManager &m_sm;
    clang::LangOptions m_lang_opts;

    // CFG specific variables
    SpmdTUTy &m_spmd_tutbl;
    pass_sequence_t pass_sequence;
    Workspace m_workspace;
};

} // namespace pass

} // namespace spmdfy
#endif