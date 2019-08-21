#ifndef FIND_SYNCTHREAD_NODES_HPP
#define FIND_SYNCTHREAD_NODES_HPP

#include <spmdfy/CFG/RecursiveCFGVisitor.hpp>
#include <clang/AST/Expr.h>
#include <spmdfy/Pass/PassHandler.hpp>

namespace spmdfy {

namespace pass {

class FindSyncthreadNodes : public cfg::RecursiveCFGVisitor<FindSyncthreadNodes> {
  public:
    FindSyncthreadNodes(SpmdTUTy &node, clang::ASTContext &ast_context,
                        Workspace &workspace)
        : m_ast_context(ast_context), m_sm(ast_context.getSourceManager()),
          m_lang_opts(ast_context.getLangOpts()), m_node(node),
          m_workspace(workspace) {
        m_lang_opts.CPlusPlus = true;
        m_lang_opts.Bool = true;
    }

#define CFGNODE_VISITOR(NODE)                                                  \
    auto Visit##NODE##Node(cfg::NODE##Node *)->bool

    CFGNODE_VISITOR(Internal);

  private:
    // AST specific variables
    clang::ASTContext &m_ast_context;
    clang::SourceManager &m_sm;
    clang::LangOptions m_lang_opts;

    cfg::CFGNode::Context m_tu_context;
    Workspace &m_workspace;
    SpmdTUTy &m_node;
};

bool findSyncthreadNodes(SpmdTUTy &, clang::ASTContext &, Workspace &);

PASS(findSyncthreadNodes, find_syncthreads_nodes_pass_t);

#undef CFGNODE_VISITOR

} // namespace pass
} // namespace spmdfy

#endif