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

GlobalVarNode::GlobalVarNode(clang::ASTContext &ast_context,
                             const clang::VarDecl *var_decl)
    : m_ast_context(ast_context) {
    m_var_decl = var_decl;
    m_name = var_decl->getNameAsString();
    m_node_type = GlobalVar;
    m_context = Global;
    SPMDFY_INFO("Creating GlobalVarNode {}", m_name);
}

// :GlobalVarNode

ForwardNode::ForwardNode() {
    m_node_type = Forward;
    m_name = getNodeTypeName();
    m_next = new CFGEdge();
}

auto ForwardNode::splitEdge(BiDirectNode *node) -> BiDirectNode * {
    SPMDFY_INFO("Splitting at {}", getName());
    // 1. Getting current's next
    auto next = m_next->getTerminal();
    if (next == nullptr) {
        SPMDFY_ERROR("[{}] Cannot Split Edge as next node is null",
                     getSource());
        return nullptr;
    }

    SPMDFY_INFO("Splitting from:");
    SPMDFY_INFO("{} -> {}", getSource(), next->getSource());
    SPMDFY_INFO("to:");

    // 2. Setting node's next to current's next
    node->setNext(next);

    // 3. Setting current's to point to node
    m_next->setTerminal(node);

    // 4. Setting back edges
    next->setPrevious(node);
    node->setPrevious(this);

    SPMDFY_INFO("{} -> {} -> {}", node->getPrevious()->getSource(),
                node->getSource(), node->getNext()->getSource());
    return node;
}

auto ForwardNode::getNext() -> CFGNode *const { return m_next->getTerminal(); }

auto ForwardNode::setNext(CFGNode *node, CFGEdge::Edge edge_type) -> CFGNode * {
    return m_next->setTerminal(node, edge_type);
}

// :ForwardNode

BackwardNode::BackwardNode() {
    m_node_type = Backward;
    m_name = getNodeTypeName();
    m_prev = new CFGEdge();
}

auto BackwardNode::getPrevious() -> CFGNode *const {
    return m_prev->getTerminal();
}

auto BackwardNode::setPrevious(CFGNode *node, CFGEdge::Edge edge_type)
    -> CFGNode * {
    return m_prev->setTerminal(node, edge_type);
}

// :BackwardNode

KernelFuncNode::KernelFuncNode(clang::ASTContext &ast_context,
                               const clang::FunctionDecl *func_decl)
    : m_ast_context(ast_context) {
    SPMDFY_INFO("Creating KerneFuncNode {}", func_decl->getNameAsString());
    m_name = func_decl->getNameAsString();
    m_source = m_name;
    m_func_decl = func_decl;
    m_node_type = KernelFunc;
    m_context = Global;
    m_exit = new CFGEdge();
}

auto KernelFuncNode::getName() -> std::string const {
    return m_func_decl->getNameAsString();
}

auto KernelFuncNode::getKernelNode() -> const clang::FunctionDecl *const {
    return m_func_decl;
}

auto KernelFuncNode::getExit() -> ExitNode *const {
    return dynamic_cast<ExitNode *>(m_exit->getTerminal());
}
auto KernelFuncNode::setExit(ExitNode *node, CFGEdge::Edge edge_type)
    -> ExitNode * {
    m_exit->setTerminal(node, edge_type);
    node->setPrevious(this, edge_type);
    return node;
}

// :KernelFuncNode

ConditionalNode::ConditionalNode(clang::ASTContext &ast_context,
                                 const clang::Stmt *stmt)
    : m_ast_context(ast_context) {
    m_node_type = Conditional;
    m_name = getNodeTypeName();
    true_b = m_next;
    reconv = new CFGEdge();
    m_cond_stmt = stmt;
}

auto ConditionalNode::getReconv() -> CFGNode *const {
    return reconv->getTerminal();
}

auto ConditionalNode::setReconv(CFGNode *node, CFGEdge::Edge edge_type)
    -> CFGNode * {
    return reconv->setTerminal(node, edge_type);
}

// :ConditionalNode

IfStmtNode::IfStmtNode(clang::ASTContext &ast_context,
                       const clang::IfStmt *if_stmt)
    : ConditionalNode(ast_context, if_stmt) {
    m_node_type = IfStmt;
    m_name = getNodeTypeName();
    false_b = new CFGEdge();
    m_source =
        "if (" +
        sourceDump(ast_context.getSourceManager(), ast_context.getLangOpts(),
                   if_stmt->getCond()->getSourceRange().getBegin(),
                   if_stmt->getCond()->getSourceRange().getEnd()) +
        ")";
}

auto IfStmtNode::splitTrueEdge(BiDirectNode *node) -> BiDirectNode * {
    SPMDFY_INFO("Splitting True Edge");
    return splitEdge(node);
}

auto IfStmtNode::splitFalseEdge(BiDirectNode *node) -> BiDirectNode * {
    SPMDFY_INFO("Splitting False Edge");
    auto next = false_b->getTerminal();
    if (next == nullptr) {
        SPMDFY_ERROR("[{}] Cannot Split Edge as next node is null",
                     getSource());
        return nullptr;
    }
    SPMDFY_INFO("Edge splitting from:");
    SPMDFY_INFO("{} -> {}", "IfStmtNode", next->getSource());
    SPMDFY_INFO("to:");
    node->setNext(next, cfg::CFGEdge::Complete);
    false_b->setTerminal(node, cfg::CFGEdge::Complete);
    next->setPrevious(node, cfg::CFGEdge::Complete);
    node->setPrevious(this, cfg::CFGEdge::Complete);
    SPMDFY_INFO("{} -> {} -> {}", node->getPrevious()->getSource(),
                node->getSource(), node->getNext()->getSource());
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

ForStmtNode::ForStmtNode(clang::ASTContext &ast_context,
                         const clang::ForStmt *for_stmt)
    : ConditionalNode(ast_context, for_stmt) {
    m_node_type = ForStmt;
    m_source =
        sourceDump(ast_context.getSourceManager(), ast_context.getLangOpts(),
                   for_stmt->getSourceRange().getBegin(),
                   for_stmt->getBody()->getSourceRange().getBegin());
}

// :ForStmtNode

ReconvNode::ReconvNode(ConditionalNode *cond_node) {
    m_node_type = Reconv;
    m_name = "ReconvNode";
    m_source = std::string();
    m_context = Kernel;
    back = m_prev;
    back->setTerminal(cond_node);
}

auto ReconvNode::setPrevious(CFGNode *node, CFGEdge::Edge edge_type)
    -> CFGNode * {
    return nullptr;
}

auto ReconvNode::setBack(CFGNode *node, CFGEdge::Edge edge_type) -> CFGNode * {
    return back->setTerminal(node, edge_type);
}

auto ReconvNode::getBack() -> CFGNode *const { return back->getTerminal(); }

// :ReconvNode

InternalNode::InternalNode(clang::ASTContext &ast_context, InternalNodeTy node)
    : m_node(node), m_ast_context(ast_context) {
    SPMDFY_INFO("Creating InternalNode {}", getInternalNodeName());
    m_node_type = Internal;
    m_name = getInternalNodeName();
}

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

ExitNode::ExitNode() {
    SPMDFY_INFO("Creating ExitNode");
    m_node_type = Exit;
    m_name = "ExitNode";
    m_source = std::string();
}

// :ExitNode

ISPCBlockNode::ISPCBlockNode() {
    SPMDFY_INFO("Creating ISPCBlockNode");
    m_node_type = ISPCBlock;
}

// :ISPCBlockNode

ISPCBlockExitNode::ISPCBlockExitNode() {
    SPMDFY_INFO("Creating ISPCBlockExitNode");
    m_node_type = ISPCBlockExit;
    m_source = "ISPC_BLOCK_END";
}

// :ISPCBlockExitNode

ISPCGridNode::ISPCGridNode() {
    SPMDFY_INFO("Creating ISPCGridNode");
    m_node_type = ISPCGrid;
    m_source = "ISPC_GRID_START";
}

// :ISPCGridNode

ISPCGridExitNode::ISPCGridExitNode() {
    SPMDFY_INFO("Creating ISPCGridExitNode");
    m_node_type = ISPCGridExit;
    m_source = "ISPC_GRID_END";
}

// :ISPCGridExitNode

auto rmCFGNode(CFGNode *node) -> cfg::CFGNode * {
    SPMDFY_INFO("Removing nodes: ");
    // 1. get next and previous of current
    auto next = node->getNext();
    auto prev = node->getPrevious();
    SPMDFY_INFO("{} -> {} -> {}", prev->getSource(), node->getSource(),
                next->getSource());

    // 2. Updating then nodes
    if (prev)
        next->setPrevious(prev, cfg::CFGEdge::Complete);
    if (next)
        prev->setNext(next, cfg::CFGEdge::Complete);

    SPMDFY_INFO("{} -> {}", prev->getSource(), next->getSource());
    return node;
}

// :utils

} // namespace cfg

} // namespace spmdfy