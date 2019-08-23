#ifndef CFGVISITOR_HPP
#define CFGVISITOR_HPP

#include <spmdfy/CFG/CFG.hpp>

#include <vector>

namespace spmdfy {

namespace cfg {

using SpmdTUTy = std::vector<cfg::CFGNode *>;
/**
 * \class CFGVisitor
 * \ingroup CFG
 *
 * \brief A Visitor interface the CFG inspired by clang's ASTVisitors. The
 * Visitors can be overriden by the inherited nodes.
 *
 * */
// FIXME: Why CRTP and Virtual at the same time???
template <typename Derived, typename RetTy> class CFGVisitor {
  public:
#define DISPATCH(NAME)                                                         \
    return static_cast<Derived *>(this)->Visit##NAME##Node(                    \
        dynamic_cast<NAME##Node *>(node))

#define FALLBACK(NAME)                                                         \
    virtual RetTy Visit##NAME##Node(NAME##Node *node) { DISPATCH(CFG); }

    virtual ~CFGVisitor() = default;

    FALLBACK(Forward);
    FALLBACK(Backward);
    FALLBACK(BiDirect);
    FALLBACK(GlobalVar);
    FALLBACK(KernelFunc);
    FALLBACK(Conditional);
    FALLBACK(IfStmt);
    FALLBACK(ForStmt);
    FALLBACK(Reconv);
    FALLBACK(Internal);
    FALLBACK(Exit);
    FALLBACK(ISPCBlock);
    FALLBACK(ISPCBlockExit);
    FALLBACK(ISPCGrid);
    FALLBACK(ISPCGridExit);

    virtual RetTy Visit(CFGNode *node) {
        // clang-format off
        switch (node->getNodeType()) {
        case CFGNode::Forward:       DISPATCH(Forward);
        case CFGNode::Backward:      DISPATCH(Backward);
        case CFGNode::BiDirect:      DISPATCH(BiDirect);
        case CFGNode::GlobalVar:     DISPATCH(GlobalVar);
        case CFGNode::KernelFunc:    DISPATCH(KernelFunc);
        case CFGNode::Conditional:   DISPATCH(Conditional);
        case CFGNode::IfStmt:        DISPATCH(IfStmt);
        case CFGNode::ForStmt:       DISPATCH(ForStmt);
        case CFGNode::Reconv:        DISPATCH(Reconv);
        case CFGNode::Internal:      DISPATCH(Internal);
        case CFGNode::Exit:          DISPATCH(Exit);
        case CFGNode::ISPCBlock:     DISPATCH(ISPCBlock);
        case CFGNode::ISPCBlockExit: DISPATCH(ISPCBlockExit);
        case CFGNode::ISPCGrid:      DISPATCH(ISPCGrid);
        case CFGNode::ISPCGridExit:  DISPATCH(ISPCGridExit);
        default:
            SPMDFY_ERROR("Unknown Node Kind");
        }
        // clang-format on
        return RetTy();
    }

    virtual RetTy VisitCFGNode(CFGNode *node) { return RetTy(); }
};

} // namespace cfg

} // namespace spmdfy

#endif