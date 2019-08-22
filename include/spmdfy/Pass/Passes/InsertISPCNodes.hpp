#ifndef INSERT_ISPC_NODES_HPP
#define INSERT_ISPC_NODES_HPP

#include <spmdfy/CFG/RecursiveCFGVisitor.hpp>
#include <spmdfy/Pass/PassHandler.hpp>

namespace spmdfy {

namespace pass {

class InsertISPCNodes : public cfg::RecursiveCFGVisitor<InsertISPCNodes> {
  public:
    using cfg::RecursiveCFGVisitor<InsertISPCNodes>::Visit;
    InsertISPCNodes(SpmdTUTy &node, clang::ASTContext &ast_context,
                   Workspace &workspace)
        : m_ast_context(ast_context), m_sm(ast_context.getSourceManager()),
          m_lang_opts(ast_context.getLangOpts()), m_node(node),
          m_workspace(workspace) {
        m_lang_opts.CPlusPlus = true;
        m_lang_opts.Bool = true;
    }

    auto VisitKernelFuncNode(cfg::KernelFuncNode *kernel) -> bool;

  private:
    // AST specific variables
    clang::ASTContext &m_ast_context;
    clang::SourceManager &m_sm;
    clang::LangOptions m_lang_opts;

    cfg::CFGNode::Context m_tu_context;
    Workspace &m_workspace;
    SpmdTUTy &m_node;
};

bool insertISPCNodes(SpmdTUTy &, clang::ASTContext &, Workspace &);

PASS(insertISPCNodes, insert_ispc_nodes_pass_t);

} // namespace pass
} // namespace spmdfy

#endif