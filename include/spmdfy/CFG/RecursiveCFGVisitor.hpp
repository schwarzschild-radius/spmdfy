#ifndef RECURSIVE_CFG_VISITOR_HPP
#define RECURSIVE_CFG_VISITOR_HPP

#include <spmdfy/CFG/CFGVisitor.hpp>
#include <spmdfy/Logger.hpp>
#include <spmdfy/utils.hpp>

#include <vector>

namespace spmdfy {
namespace cfg {

using SpmdTUTy = std::vector<CFGNode *>;

template <typename Derived>
class RecursiveCFGVisitor
    : public CFGVisitor<RecursiveCFGVisitor<Derived>, bool> {
#define ISNODE(NODE, TYPE) (NODE->getNodeType() == TYPE)

#define VISIT(NODE) static_cast<Derived *>(this)->Visit(NODE)
  public:
    using CFGVisitor<RecursiveCFGVisitor<Derived>, bool>::Visit;
    bool HandleSpmdTU(SpmdTUTy &tu) {
        for (auto node : tu) {
            Visit(node);
        }
        return true;
    }

    bool VisitKernelFuncNode(KernelFuncNode *node) {
        SPMDFY_INFO("Recursively Visiting KernelFunc");
        for (CFGNode *curr_node = node->getNext();
             !ISNODE(curr_node, CFGNode::Exit);
             curr_node = curr_node->getNext()) {
            VISIT(curr_node);
        }
        return true;
    }

    bool VisitIfStmtNode(IfStmtNode *node) {
        SPMDFY_INFO("Recursively Visiting IfStmt");
        // 1. Visiting true
        for (auto curr_node = node->getTrueBlock();
             !ISNODE(curr_node, CFGNode::Reconv);
             curr_node = curr_node->getNext()) {
            VISIT(curr_node);
        }

        // 2. Visiting false
        for (auto curr_node = node->getFalseBlock();
             !ISNODE(curr_node, CFGNode::Reconv);
             curr_node = curr_node->getNext()) {
            VISIT(curr_node);
        }
        return true;
    }

    bool VisitForStmtNode(ForStmtNode *node) {
        SPMDFY_INFO("Recursively Visiting ForStmt");
        for (auto curr_node = node->getNext();
             !(ISNODE(curr_node, CFGNode::Reconv));
             curr_node = curr_node->getNext()) {
            VISIT(curr_node);
        }
        return true;
    }
};
} // namespace cfg

} // namespace spmdfy

#endif