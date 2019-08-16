#include <spmdfy/Generator/CFGGenerator/CFG.hpp>

namespace spmdfy {

namespace CFG {

auto CFGNode::getNodeTypeName() -> std::string const {
    switch (m_node_type) {
    case GlobalVar:
        return "GlobalVarNode";
    case KernelFunc:
        return "KernelFuncNode";
    case Internal:
        return "InternalNode";
    case Exit:
        return "ExitNode";
    default:
        return "CFGNode";
    }
}

auto CFGNode::getContextTypeName() -> std::string const {
    switch (m_context) {
    case Global:
        return "GlobalContext";
    case Kernel:
        return "KernelContext";
    case Device:
        return "DeviceContext";
    }
}

auto CFGEdge::getEdgeTypeName() -> std::string const {
    switch (m_edge) {
    case Partial:
        return "Partial";
    case Complete:
        return "Complete";
    }
}

auto CFGEdge::getTerminal() -> CFGNode *const { return m_terminal; }

auto CFGEdge::setTerminal(CFGNode *terminal, Edge edge_type) -> bool {
    m_terminal = terminal;
    m_edge = edge_type;
    return false;
}

auto CFGEdge::setEdgeType(Edge edge_type) -> bool {
    m_edge = edge_type;
    return false;
}
// CFGEdge

auto KernelFuncNode::getNext() -> CFGNode* const {
    return next->getTerminal();
}

auto KernelFuncNode::setNext(CFGNode *node, CFGEdge::Edge edge_type) -> bool {
    if (next->setTerminal(node, edge_type))
        return true;
    SPMDFY_INFO("[KernelFunc/{}] setting edge to {} with edge type {}",
                m_func_decl->getNameAsString(), node->getNodeTypeName(),
                next->getEdgeTypeName());
    return false;
}

auto KernelFuncNode::getDeclKindString() -> std::string const {
    // FIXME: Problem with Namelookup
    std::string temp = ((const clang::Decl *)m_func_decl)->getDeclKindName();
    return temp;
}

// KernelFuncNode
auto InternalNode::getNext() -> CFGNode *const { return next->getTerminal(); }

auto InternalNode::getPrevious() -> CFGNode *const {
    return prev->getTerminal();
}

auto InternalNode::setNext(CFGNode *node, CFGEdge::Edge edge_type) -> bool {
    return next->setTerminal(node, edge_type);
}

auto InternalNode::setPrevious(CFGNode *node, CFGEdge::Edge edge_type) -> bool {
    return prev->setTerminal(node, edge_type);
}

auto InternalNode::getInternalNodeName() -> std::string const {
    return std::visit(
        visitor{
            [](const clang::Decl *decl) { return decl->getDeclKindName(); },
            [](const clang::Stmt *stmt) { return stmt->getStmtClassName(); },
            [](const clang::Expr *expr) { return expr->getStmtClassName(); },
            [](const clang::Type *type) { return type->getTypeClassName(); }},
        m_node);
}
// InternalNode

auto ExitNode::getPrevious() -> CFGNode *const { return prev->getTerminal(); }

auto ExitNode::setPrevious(CFGNode *node, CFGEdge::Edge edge_type) -> bool {
    return prev->setTerminal(node, edge_type);
}

// ExitNode

} // namespace CFG

} // namespace spmdfy