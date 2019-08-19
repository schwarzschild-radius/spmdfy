#include <spmdfy/CFG/CFG.hpp>

namespace spmdfy {

namespace cfg {

auto CFGNode::getNodeTypeName() -> std::string const {
    switch (m_node_type) {
    case GlobalVar:
        return "GlobalVarNode";
    case KernelFunc:
        return "KernelFuncNode";
    case Internal:
        return "InternalNode";
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

auto CFGEdge::getTerminal() -> CFGNode *const {
    // SPMDFY_INFO("Getting terminal node {}", m_terminal->getName());
    return m_terminal;
}

auto CFGEdge::setTerminal(CFGNode *terminal, Edge edge_type) -> bool {
    m_terminal = terminal;
    m_edge = edge_type;
    return false;
}

auto CFGEdge::setEdgeType(Edge edge_type) -> bool {
    m_edge = edge_type;
    return false;
}

auto CFGNode::getName() -> std::string const { return "No Name"; }

auto CFGNode::splitEdge(CFGNode *) -> bool {
    SPMDFY_ERROR("splitEdge supported operation for {}", getNodeTypeName());
    return true;
}

auto CFGNode::getNext() -> CFGNode *const {
    SPMDFY_ERROR("getNext supported operation for {}", getNodeTypeName());
    return nullptr;
}
auto CFGNode::getPrevious() -> CFGNode *const {
    SPMDFY_ERROR("getPrevious supported operation for {}", getNodeTypeName());
    return nullptr;
}
auto CFGNode::setNext(CFGNode *node, CFGEdge::Edge edge_type) -> bool {
    SPMDFY_ERROR("setNext supported operation for {}", getNodeTypeName());
    return true;
}
auto CFGNode::setPrevious(CFGNode *node, CFGEdge::Edge edge_type) -> bool {
    SPMDFY_ERROR("setPrevious supported operation for {}", getNodeTypeName());
    return true;
}

// CFGEdge

auto KernelFuncNode::getNext() -> CFGNode *const { return next->getTerminal(); }

auto KernelFuncNode::setNext(CFGNode *node, CFGEdge::Edge edge_type) -> bool {
    if (next->setTerminal(node, edge_type)) {
        SPMDFY_ERROR("Unable to set the node!");
        return true;
    }
    SPMDFY_INFO("[KernelFunc/{}] setting edge to {} with edge type {}",
                getName(), next->getTerminal()->getName(),
                next->getEdgeTypeName());
    return false;
}

auto KernelFuncNode::getDeclKindString() -> std::string const {
    // FIXME: Problem with Namelookup
    std::string temp = ((const clang::Decl *)m_func_decl)->getDeclKindName();
    return temp;
}

auto KernelFuncNode::splitEdge(CFGNode *node) -> bool {
    auto nxt = next->getTerminal();
    if (nxt == nullptr) {
        SPMDFY_ERROR("[{}] Cannot Split Edge as next node is null", getName());
        return true;
    }
    SPMDFY_INFO("Edge splitting from:");
    SPMDFY_INFO("{} -> {}", getName(), nxt->getName());
    SPMDFY_INFO("to:");
    node->setNext(nxt, cfg::CFGEdge::Complete);
    next->setTerminal(node, cfg::CFGEdge::Complete);
    nxt->setPrevious(node, cfg::CFGEdge::Complete);
    node->setPrevious(this, cfg::CFGEdge::Complete);
    SPMDFY_INFO("{} -> {} -> {}", node->getPrevious()->getName(),
                node->getName(), node->getNext()->getName());
    return false;
}

// KernelFuncNode

auto IfStmtNode::getName() -> std::string const {
    auto &m_sm = m_ast_context.getSourceManager();
    auto &m_lang_opts = m_ast_context.getLangOpts();
    return SRCDUMP(m_if_stmt);
}
auto IfStmtNode::splitEdge(CFGNode *node) -> bool {
    return splitTrueEdge(node);
}
auto IfStmtNode::getNext() -> CFGNode *const { return true_b->getTerminal(); }
auto IfStmtNode::getPrevious() -> CFGNode *const { return prev->getTerminal(); }
auto IfStmtNode::setNext(CFGNode *node, CFGEdge::Edge edge_type) -> bool {
    return setTrueBlock(node, edge_type);
}
auto IfStmtNode::setPrevious(CFGNode *node, CFGEdge::Edge edge_type) -> bool {
    return prev->setTerminal(node, edge_type);
}

auto IfStmtNode::getReconv() -> CFGNode * const { return reconv->getTerminal(); };

auto IfStmtNode::setReconv(CFGNode * node, CFGEdge::Edge edge_type) -> bool{
    return reconv->setTerminal(node, edge_type);
}

auto IfStmtNode::splitTrueEdge(CFGNode *node) -> bool {
    SPMDFY_INFO("Splitting True Edge");
    auto nxt = true_b->getTerminal();
    if (nxt == nullptr) {
        SPMDFY_ERROR("[{}] Cannot Split Edge as next node is null", getName());
        return true;
    }
    SPMDFY_INFO("Edge splitting from:");
    SPMDFY_INFO("{} -> {}", "IfStmtNode", nxt->getName());
    SPMDFY_INFO("to:");
    node->setNext(nxt, cfg::CFGEdge::Complete);
    true_b->setTerminal(node, cfg::CFGEdge::Complete);
    nxt->setPrevious(node, cfg::CFGEdge::Complete);
    node->setPrevious(this, cfg::CFGEdge::Complete);
    SPMDFY_INFO("{} -> {} -> {}", node->getPrevious()->getName(),
                node->getName(), node->getNext()->getName());
    return false;
}
auto IfStmtNode::splitFalseEdge(CFGNode *node) -> bool {
    SPMDFY_INFO("Splitting False Edge");
    auto nxt = false_b->getTerminal();
    if (nxt == nullptr) {
        SPMDFY_ERROR("[{}] Cannot Split Edge as next node is null", getName());
        return true;
    }
    SPMDFY_INFO("Edge splitting from:");
    SPMDFY_INFO("{} -> {}", "IfStmtNode", nxt->getName());
    SPMDFY_INFO("to:");
    node->setNext(nxt, cfg::CFGEdge::Complete);
    false_b->setTerminal(node, cfg::CFGEdge::Complete);
    nxt->setPrevious(node, cfg::CFGEdge::Complete);
    node->setPrevious(this, cfg::CFGEdge::Complete);
    SPMDFY_INFO("{} -> {} -> {}", node->getPrevious()->getName(),
                node->getName(), node->getNext()->getName());
    return false;
}

auto IfStmtNode::getTrueBlock() -> CFGNode* const{
    return true_b->getTerminal();
}
auto IfStmtNode::getFalseBlock() -> CFGNode* const{
    return false_b->getTerminal();
}

auto IfStmtNode::setTrueBlock(CFGNode *node, CFGEdge::Edge edge_type) -> bool {
    return true_b->setTerminal(node, edge_type);
}
auto IfStmtNode::setFalseBlock(CFGNode *node, CFGEdge::Edge edge_type) -> bool {
    return false_b->setTerminal(node, edge_type);
}

// IfStmtNode

auto ForStmtNode::getName() -> std::string const{
    auto &m_sm = m_ast_context.getSourceManager();
    auto &m_lang_opts = m_ast_context.getLangOpts();
    return SRCDUMP(m_for_stmt);
}
auto ForStmtNode::splitEdge(CFGNode * node) -> bool{
    SPMDFY_INFO("Splitting True Edge");
    auto nxt = true_b->getTerminal();
    if (nxt == nullptr) {
        SPMDFY_ERROR("[{}] Cannot Split Edge as next node is null", getName());
        return true;
    }
    SPMDFY_INFO("Edge splitting from:");
    SPMDFY_INFO("{} -> {}", "ForStmtNode", nxt->getName());
    SPMDFY_INFO("to:");
    node->setNext(nxt, cfg::CFGEdge::Complete);
    true_b->setTerminal(node, cfg::CFGEdge::Complete);
    nxt->setPrevious(node, cfg::CFGEdge::Complete);
    node->setPrevious(this, cfg::CFGEdge::Complete);
    SPMDFY_INFO("{} -> {} -> {}", node->getPrevious()->getName(),
                node->getName(), node->getNext()->getName());
    return false;
}

auto ForStmtNode::getNext() -> CFGNode *const{
    return true_b->getTerminal();
}
auto ForStmtNode::getPrevious() -> CFGNode *const{
    return prev->getTerminal();
}
auto ForStmtNode::setNext(CFGNode *node, CFGEdge::Edge edge_type) -> bool{
    return true_b->setTerminal(node, edge_type);
}
auto ForStmtNode::setPrevious(CFGNode *node, CFGEdge::Edge edge_type) -> bool{
    return prev->setTerminal(node, edge_type);
}
auto ForStmtNode::getReconv() -> CFGNode * const{
    return reconv->getTerminal();
}
auto ForStmtNode::setReconv(CFGNode * node, CFGEdge::Edge edge_type) -> bool{
    return reconv->setTerminal(node, edge_type);
}

// ForStmtNode

auto ReconvNode::getName() -> std::string const { return "ReconvNode"; }

auto ReconvNode::splitEdge(CFGNode *node) -> bool {
    auto nxt = next->getTerminal();
    if (nxt == nullptr) {
        SPMDFY_ERROR("[{}] Cannot Split Edge as next node is null", getName());
        return true;
    }
    SPMDFY_INFO("Edge splitting from:");
    SPMDFY_INFO("{} -> {}", getName(), nxt->getName());
    SPMDFY_INFO("to:");
    node->setNext(nxt, cfg::CFGEdge::Complete);
    next->setTerminal(node, cfg::CFGEdge::Complete);
    nxt->setPrevious(node, cfg::CFGEdge::Complete);
    node->setPrevious(this, cfg::CFGEdge::Complete);
    SPMDFY_INFO("{} -> {} -> {}", node->getPrevious()->getName(),
                node->getName(), node->getNext()->getName());
    return false;
}

auto ReconvNode::getNext() -> CFGNode *const { return next->getTerminal(); }

auto ReconvNode::getPrevious() -> CFGNode *const { return back->getTerminal(); }

auto ReconvNode::setNext(CFGNode *node, CFGEdge::Edge edge_type) -> bool {
    return next->setTerminal(node, edge_type);
}

auto ReconvNode::setPrevious(CFGNode *node, CFGEdge::Edge edge_type) -> bool {
    return false;
}

auto ReconvNode::setBack(CFGNode *node, CFGEdge::Edge edge_type) -> bool {
    return back->setTerminal(node, edge_type);
}

auto ReconvNode::getBack() -> CFGNode *const { return back->getTerminal(); }

// ReconvNode

auto InternalNode::getNext() -> CFGNode *const { return next->getTerminal(); }

auto InternalNode::getPrevious() -> CFGNode *const {
    return prev->getTerminal();
}

auto InternalNode::setNext(CFGNode *node, CFGEdge::Edge edge_type) -> bool {
    if (next->setTerminal(node, edge_type)) {
        return true;
    }
    SPMDFY_INFO("[Internal/{}] setting edge to {} with edge type {}", getName(),
                next->getTerminal()->getName(), next->getEdgeTypeName());
    return false;
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

auto InternalNode::getName() -> std::string const {
    auto &m_sm = m_ast_context.getSourceManager();
    auto m_lang_opts = m_ast_context.getLangOpts();
    m_lang_opts.CPlusPlus = true;
    m_lang_opts.Bool = true;
    return std::visit(
        visitor{[](const clang::Decl *decl) -> std::string {
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

auto InternalNode::splitEdge(CFGNode *node) -> bool {
    auto nxt = next->getTerminal();
    if (nxt == nullptr) {
        SPMDFY_ERROR("[{}] Cannot Split Edge as next node is null", getName());
        return true;
    }
    SPMDFY_INFO("Edge splitting from:");
    SPMDFY_INFO("{} -> {}", getName(), nxt->getName());
    SPMDFY_INFO("to:");
    node->setNext(nxt, cfg::CFGEdge::Complete);
    next->setTerminal(node, cfg::CFGEdge::Complete);
    nxt->setPrevious(node, cfg::CFGEdge::Complete);
    node->setPrevious(this, cfg::CFGEdge::Complete);
    SPMDFY_INFO("{} -> {} -> {}", node->getPrevious()->getName(),
                node->getName(), node->getNext()->getName());
    return false;
}

// InternalNode

auto ExitNode::getPrevious() -> CFGNode *const { return prev->getTerminal(); }

auto ExitNode::setPrevious(CFGNode *node, CFGEdge::Edge edge_type) -> bool {
    return prev->setTerminal(node, edge_type);
}

// ExitNode

auto ISPCBlockNode::getName() -> std::string const { return "ISPCBlock"; }

auto ISPCBlockNode::getNext() -> CFGNode *const { return next->getTerminal(); }

auto ISPCBlockNode::getPrevious() -> CFGNode *const {
    return prev->getTerminal();
}

auto ISPCBlockNode::setNext(CFGNode *node, CFGEdge::Edge edge_type) -> bool {
    if (next->setTerminal(node, edge_type)) {
        return true;
    }

    SPMDFY_INFO("[{}] setting edge to {} with edge type {}", getName(),
                next->getTerminal()->getName(), next->getEdgeTypeName());
    return false;
}
auto ISPCBlockNode::setPrevious(CFGNode *node, CFGEdge::Edge edge_type)
    -> bool {
    if (prev->setTerminal(node, edge_type)) {
        return true;
    }

    SPMDFY_INFO("[{}] setting edge to {} with edge type {}", getName(),
                prev->getTerminal()->getName(), prev->getEdgeTypeName());
    return false;
}

// ISPCBlockNode

auto ISPCBlockExitNode::getName() -> std::string const {
    return "ISPCBlockExit";
}
auto ISPCBlockExitNode::getNext() -> CFGNode *const {
    return next->getTerminal();
}
auto ISPCBlockExitNode::getPrevious() -> CFGNode *const {
    return prev->getTerminal();
}
auto ISPCBlockExitNode::setNext(CFGNode *node, CFGEdge::Edge edge_type)
    -> bool {
    if (next->setTerminal(node, edge_type)) {
        return true;
    }

    SPMDFY_INFO("[{}] setting edge to {} with edge type {}", getName(),
                next->getTerminal()->getName(), next->getEdgeTypeName());
    return false;
}
auto ISPCBlockExitNode::setPrevious(CFGNode *node, CFGEdge::Edge edge_type)
    -> bool {
    if (prev->setTerminal(node, edge_type)) {
        return true;
    }

    SPMDFY_INFO("[{}] setting edge to {} with edge type {}", getName(),
                prev->getTerminal()->getName(), prev->getEdgeTypeName());
    return false;
}

// ISPCBlockExitNode

auto ISPCGridNode::getName() -> std::string const { return "ISPCGrid"; }

auto ISPCGridNode::getNext() -> CFGNode *const { return next->getTerminal(); }

auto ISPCGridNode::getPrevious() -> CFGNode *const {
    return prev->getTerminal();
}

auto ISPCGridNode::setNext(CFGNode *node, CFGEdge::Edge edge_type) -> bool {
    if (next->setTerminal(node, edge_type)) {
        return true;
    }

    SPMDFY_INFO("[{}] setting edge to {} with edge type {}", getName(),
                next->getTerminal()->getName(), next->getEdgeTypeName());
    return false;
}
auto ISPCGridNode::setPrevious(CFGNode *node, CFGEdge::Edge edge_type) -> bool {
    if (prev->setTerminal(node, edge_type)) {
        return true;
    }

    SPMDFY_INFO("[{}] setting edge to {} with edge type {}", getName(),
                prev->getTerminal()->getName(), prev->getEdgeTypeName());
    return false;
}

// ISPCGridNode

auto ISPCGridExitNode::getName() -> std::string const { return "ISPCGridExit"; }
auto ISPCGridExitNode::getNext() -> CFGNode *const {
    return next->getTerminal();
}
auto ISPCGridExitNode::getPrevious() -> CFGNode *const {
    return prev->getTerminal();
}
auto ISPCGridExitNode::setNext(CFGNode *node, CFGEdge::Edge edge_type) -> bool {
    if (next->setTerminal(node, edge_type)) {
        return true;
    }

    SPMDFY_INFO("[{}] setting edge to {} with edge type {}", getName(),
                next->getTerminal()->getName(), next->getEdgeTypeName());
    return false;
}
auto ISPCGridExitNode::setPrevious(CFGNode *node, CFGEdge::Edge edge_type)
    -> bool {
    if (prev->setTerminal(node, edge_type)) {
        return true;
    }

    SPMDFY_INFO("[{}] setting edge to {} with edge type {}", getName(),
                prev->getTerminal()->getName(), prev->getEdgeTypeName());
    return false;
}

// ISPCGridExitNode

} // namespace cfg

} // namespace spmdfy