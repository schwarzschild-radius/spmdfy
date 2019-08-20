#ifndef CFGVISITOR_HPP
#define CFGVISITOR_HPP

#include <spmdfy/CFG/CFG.hpp>

#include <vector>

namespace spmdfy {

namespace cfg {

using SpmdTUTy = std::vector<cfg::CFGNode *>;

template <typename Derived, typename RetTy> class CFGVisitor {
  public:
#define DISPATCH(NAME)                                                         \
    return static_cast<Derived *>(this)->Visit##NAME##Node(                    \
        dynamic_cast<NAME##Node *>(node))

#define FALLBACK(NAME)                                                         \
    RetTy Visit##NAME##Node(NAME##Node *node) { DISPATCH(CFG); }

    virtual ~CFGVisitor() = default;

    FALLBACK(GlobalVar);
    FALLBACK(KernelFunc);
    FALLBACK(IfStmt);
    FALLBACK(ForStmt);
    FALLBACK(Reconv);
    FALLBACK(Internal);
    FALLBACK(Exit);
    FALLBACK(ISPCBlock);
    FALLBACK(ISPCBlockExit);
    FALLBACK(ISPCGrid);
    FALLBACK(ISPCGridExit);

    RetTy Visit(CFGNode *node) {
        // clang-format off
        switch (node->getNodeType()) {
        case CFGNode::GlobalVar:     DISPATCH(GlobalVar);
        case CFGNode::KernelFunc:    DISPATCH(KernelFunc);
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

    RetTy VisitCFGNode(CFGNode *node) { return RetTy(); }
};

} // namespace cfg

} // namespace spmdfy

#endif