#ifndef DETECT_PARTIAL_NODES_HPP
#define DETECT_PARTIAL_NODES_HPP

#include <clang/AST/Expr.h>
#include <spmdfy/CFG/RecursiveCFGVisitor.hpp>
#include <spmdfy/Pass/PassHandler.hpp>

namespace spmdfy {

namespace pass {

class DetectPartialNodes
    : public cfg::RecursiveCFGVisitor<DetectPartialNodes> {
  public:
    DetectPartialNodes(SpmdTUTy &node, clang::ASTContext &ast_context,
                          Workspace &workspace)
        : m_ast_context(ast_context), m_sm(ast_context.getSourceManager()),
          m_lang_opts(ast_context.getLangOpts()), m_node(node),
          m_workspace(workspace) {
        m_lang_opts.CPlusPlus = true;
        m_lang_opts.Bool = true;
    }
#define CFGNODE_VISITOR(NODE) auto Visit##NODE##Node(cfg::NODE##Node *)->bool

    CFGNODE_VISITOR(KernelFunc);
    CFGNODE_VISITOR(ForStmt);

  private:
    // AST specific variables
    clang::ASTContext &m_ast_context;
    clang::SourceManager &m_sm;
    clang::LangOptions m_lang_opts;

    cfg::CFGNode::Context m_tu_context;
    Workspace &m_workspace;
    SpmdTUTy &m_node;
};

bool detectPartialNodes(SpmdTUTy &, clang::ASTContext &, Workspace &);

PASS(detectPartialNodes, detect_partial_nodes_pass_t);

#undef CFGNODE_VISITOR

} // namespace pass
} // namespace spmdfy

#endif