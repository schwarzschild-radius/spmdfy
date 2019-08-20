#include <spmdfy/CFG/CFG.hpp>

namespace spmdfy {

namespace cfg {

auto CFGNode::getNodeTypeName() -> std::string const {
    switch (m_node_type) {
    case Forward:
        return "ForwardNode";
    case Backward:
        return "BackwardNode";
    case BiDirect:
        return "BiDirectNode";
    case GlobalVar:
        return "GlobalVarNode";
    case KernelFunc:
        return "KernelFuncNode";
    case Internal:
        return "InternalNode";
    case Conditional:
        return "ConditionalNode";
    case IfStmt:
        return "IfStmtNode";
    case ForStmt:
        return "ForStmtNode";
    case Reconv:
        return "ReconvNode";
    case Exit:
        return "ExitNode";
    case ISPCBlock:
        return "ISPCBlockNode";
    case ISPCBlockExit:
        return "ISPCBlockExitNode";
    case ISPCGrid:
        return "ISPCGridNode";
    case ISPCGridExit:
        return "ISPCGridExitNode";
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

// :EnumString

auto CFGEdge::getTerminal() -> CFGNode *const { return m_terminal; }

auto CFGEdge::setTerminal(CFGNode *terminal, Edge edge_type) -> CFGNode * {
    m_terminal = terminal;
    m_edge = edge_type;
    return terminal;
}

auto CFGEdge::setEdgeType(Edge edge_type) -> Edge {
    m_edge = edge_type;
    return edge_type;
}

// :CFGEdge

auto CFGNode::getSource() -> std::string const { return m_source; }
auto CFGNode::setSource(const std::string &source) -> std::string {
    return (m_source = source);
}
auto CFGNode::getName() -> std::string const { return m_name; }

auto CFGNode::splitEdge(BiDirectNode *) -> BiDirectNode * {
    SPMDFY_ERROR("splitEdge supported operation for {}", getNodeTypeName());
    return nullptr;
}

auto CFGNode::getNext() -> CFGNode *const {
    SPMDFY_ERROR("getNext supported operation for {}", getNodeTypeName());
    return nullptr;
}
auto CFGNode::setNext(CFGNode *node, CFGEdge::Edge edge_type) -> CFGNode * {
    SPMDFY_ERROR("setNext supported operation for {}", getNodeTypeName());
    return nullptr;
}
auto CFGNode::getPrevious() -> CFGNode *const {
    SPMDFY_ERROR("getPrevious supported operation for {}", getNodeTypeName());
    return nullptr;
}
auto CFGNode::setPrevious(CFGNode *node, CFGEdge::Edge edge_type) -> CFGNode * {
    SPMDFY_ERROR("setPrevious supported operation for {}", getNodeTypeName());
    return nullptr;
}

// :CFGNode

auto ForwardNode::splitEdge(BiDirectNode *node) -> BiDirectNode * {
    SPMDFY_INFO("Splitting at forward");
    // 1. Getting current's next
    auto next = m_next->getTerminal();
    if (next == nullptr) {
        SPMDFY_ERROR("[{}] Cannot Split Edge as next node is null", getName());
        return nullptr;
    }

    SPMDFY_INFO("Splitting from:");
    SPMDFY_INFO("{} -> {}", getName(), next->getName());
    SPMDFY_INFO("to:");

    // 2. Setting node's next to current's next
    node->setNext(next);

    // 3. Setting current's to point to node
    m_next->setTerminal(node);

    // 4. Setting back edges
    next->setPrevious(node);
    node->setPrevious(this);

    SPMDFY_INFO("{} -> {} -> {}", node->getPrevious()->getName(),
                node->getName(), node->getNext()->getName());
    return node;
}

auto ForwardNode::getNext() -> CFGNode *const { return m_next->getTerminal(); }

auto ForwardNode::setNext(CFGNode *node, CFGEdge::Edge edge_type) -> CFGNode * {
    return m_next->setTerminal(node, edge_type);
}

// :ForwardNode

auto BackwardNode::getPrevious() -> CFGNode *const{
    return m_prev->getTerminal();
}

auto BackwardNode::setPrevious(CFGNode *node,
                               CFGEdge::Edge edge_type)
    -> CFGNode * {
    return m_prev->setTerminal(node, edge_type);
}

// :BackwardNode

auto KernelFuncNode::getName() -> std::string const {
    return m_func_decl->getNameAsString();
}

auto KernelFuncNode::getKernelNode() -> const clang::FunctionDecl *const {
    return m_func_decl;
}

// :KernelFuncNode

auto ConditionalNode::getReconv() -> CFGNode *const {
    return reconv->getTerminal();
}

auto ConditionalNode::setReconv(CFGNode *node, CFGEdge::Edge edge_type)
    -> CFGNode * {
    return reconv->setTerminal(node, edge_type);
}

// :ConditionalNode

auto IfStmtNode::splitTrueEdge(BiDirectNode *node) -> BiDirectNode * {
    SPMDFY_INFO("Splitting True Edge");
    return splitEdge(node);
}

auto IfStmtNode::splitFalseEdge(BiDirectNode *node) -> BiDirectNode * {
    SPMDFY_INFO("Splitting False Edge");
    auto next = false_b->getTerminal();
    if (next == nullptr) {
        SPMDFY_ERROR("[{}] Cannot Split Edge as next node is null", getName());
        return nullptr;
    }
    SPMDFY_INFO("Edge splitting from:");
    SPMDFY_INFO("{} -> {}", "IfStmtNode", next->getName());
    SPMDFY_INFO("to:");
    node->setNext(next, cfg::CFGEdge::Complete);
    false_b->setTerminal(node, cfg::CFGEdge::Complete);
    next->setPrevious(node, cfg::CFGEdge::Complete);
    node->setPrevious(this, cfg::CFGEdge::Complete);
    SPMDFY_INFO("{} -> {} -> {}", node->getPrevious()->getName(),
                node->getName(), node->getNext()->getName());
    return node;
}

auto IfStmtNode::getTrueBlock() -> CFGNode *const {
    return true_b->getTerminal();
}
auto IfStmtNode::getFalseBlock() -> CFGNode *const {
    return false_b->getTerminal();
}

auto IfStmtNode::setTrueBlock(CFGNode *node, CFGEdge::Edge edge_type)
    -> CFGNode * {
    return true_b->setTerminal(node, edge_type);
}
auto IfStmtNode::setFalseBlock(CFGNode *node, CFGEdge::Edge edge_type)
    -> CFGNode * {
    return false_b->setTerminal(node, edge_type);
}

// :IfStmtNode

// :ForStmtNode

auto ReconvNode::setPrevious(CFGNode *node, CFGEdge::Edge edge_type)
    -> CFGNode * {
    return nullptr;
}

auto ReconvNode::setBack(CFGNode *node, CFGEdge::Edge edge_type) -> CFGNode * {
    return back->setTerminal(node, edge_type);
}

auto ReconvNode::getBack() -> CFGNode *const { return back->getTerminal(); }

// :ReconvNode

auto InternalNode::getInternalNodeName() -> std::string const {
    return std::visit(
        Overload{
            [](const clang::Decl *decl) { return decl->getDeclKindName(); },
            [](const clang::Stmt *stmt) { return stmt->getStmtClassName(); },
            [](const clang::Expr *expr) { return expr->getStmtClassName(); },
            [](const clang::Type *type) { return type->getTypeClassName(); }},
        m_node);
}

auto InternalNode::getInternalNode() -> InternalNodeTy const { return m_node; }

auto InternalNode::getSource() -> std::string const {
    auto &m_sm = m_ast_context.getSourceManager();
    auto m_lang_opts = m_ast_context.getLangOpts();
    m_lang_opts.CPlusPlus = true;
    m_lang_opts.Bool = true;
    return std::visit(
        Overload{[](const clang::Decl *decl) -> std::string {
                     if (llvm::isa<const clang::NamedDecl>(decl)) {
                         return llvm::cast<const clang::NamedDecl>(decl)
                             ->getNameAsString();
                     }
                     return "";
                 },
                 [&m_sm, &m_lang_opts](const clang::Stmt *stmt) -> std::string {
                     return SRCDUMP(stmt);
                 },
                 [&m_sm, &m_lang_opts](const clang::Expr *expr) -> std::string {
                     return SRCDUMP(expr);
                 },
                 [&m_lang_opts](const clang::Type *type) -> std::string {
                     clang::PrintingPolicy pm(m_lang_opts);
                     clang::QualType qual =
                         clang::QualType::getFromOpaquePtr(type);
                     return qual.getAsString(pm);
                 }},
        m_node);
}

// :InternalNode

// :ExitNode

} // namespace cfg

} // namespace spmdfy