#include <spmdfy/Generator/CFGGenerator/ConstructCFG.hpp>

namespace spmdfy {

#define DEF_CFG_VISITOR(NODE, BASE, PARAM)                                     \
    auto ConstructSpmdCFG::Visit##NODE##BASE(clang::NODE##BASE *PARAM)->bool

#define STMT_COUNT(STMT, STMT_CLASS)                                           \
    SPMDFY_INFO("s{} {} {}", m_stmt_count, STMT, STMT_CLASS);                  \
    m_stmt_count++;

DEF_CFG_VISITOR(Compound, Stmt, cpmd) {
    for (auto stmt : cpmd->body()) {
        if (!TraverseStmt(stmt))
            continue;
        STMT_COUNT(SRCDUMP(stmt), stmt->getStmtClassName());
    }
    return false;
}

DEF_CFG_VISITOR(Decl, Stmt, decl_stmt) {
    for (auto decl : decl_stmt->decls()) {
        STMT_COUNT(SRCDUMP(decl), decl_stmt->getStmtClassName());
        cfg::InternalNode *decl_node = new cfg::InternalNode(
            m_context, llvm::cast<const clang::VarDecl>(decl));
        splitEdge(decl_node);
        break;
    }
    return false;
}

DEF_CFG_VISITOR(For, Stmt, for_stmt) {
    STMT_COUNT(sourceDump(m_sm, m_lang_opts, for_stmt->getForLoc(),
                          for_stmt->getRParenLoc()),
               for_stmt->getStmtClassName());

    // 1. Create for node
    cfg::ForStmtNode *for_node = new cfg::ForStmtNode(m_context, for_stmt);

    // 2. Inserting for node
    m_curr_node->splitEdge(for_node);
    m_curr_node = for_node;

    // 3. Creating reconv node
    cfg::ReconvNode *reconv = new cfg::ReconvNode();
    reconv->setBack(for_node, cfg::CFGEdge::Complete);

    // 4. Setting for's True to point to reconv
    for_node->splitEdge(reconv);

    // 5. Setting for to point to reconv
    for_node->setReconv(reconv, cfg::CFGEdge::Complete);

    TraverseStmt(for_stmt->getBody());
    STMT_COUNT("Reconv }", "ReconvNode");
    
    // 6. Setting reconv as current
    m_curr_node = reconv;

    return false;
}

DEF_CFG_VISITOR(If, Stmt, if_stmt) {
    STMT_COUNT(sourceDump(m_sm, m_lang_opts, if_stmt->getBeginLoc(),
                          if_stmt->getCond()->getEndLoc()),
               if_stmt->getStmtClassName());
    // 1. Creating if node
    cfg::IfStmtNode *if_node = new cfg::IfStmtNode(m_context, if_stmt);

    // 2. Inserting if node
    m_curr_node->splitEdge(if_node);
    m_curr_node = if_node;

    // 3. Creating reconv node
    cfg::ReconvNode *reconv = new cfg::ReconvNode();
    reconv->setBack(if_node, cfg::CFGEdge::Complete);

    // 4. Setting if's True to point to reconv
    if_node->splitEdge(reconv);

    // 5. Setting if's False to point to reconv
    if_node->setFalseBlock(reconv, cfg::CFGEdge::Complete);

    // 6. Setting if to point to reconv
    if_node->setReconv(reconv, cfg::CFGEdge::Complete);

    if (if_stmt->getThen()) {
        TraverseStmt(if_stmt->getThen());
    }

    if (if_stmt->getElse()) {
        TraverseStmt(if_stmt->getElse());
    }

    STMT_COUNT("Reconv }", "ReconvNode");

    // 7. Setting current node as Reconv
    m_curr_node = reconv;
    return false;
}

DEF_CFG_VISITOR(Call, Expr, call) {
    STMT_COUNT(SRCDUMP(call), call->getStmtClassName());
    cfg::InternalNode *call_node = new cfg::InternalNode(m_context, call);
    m_curr_node->splitEdge(call_node);
    m_curr_node = call_node;
    return false;
}

DEF_CFG_VISITOR(PseudoObject, Expr, pseudo) { return false; }

DEF_CFG_VISITOR(CompoundAssign, Operator, assgn) {
    STMT_COUNT(SRCDUMP(assgn), assgn->getStmtClassName());
    cfg::InternalNode *assgn_node = new cfg::InternalNode(m_context, assgn);
    m_curr_node->splitEdge(assgn_node);
    m_curr_node = assgn_node;
    return false;
}

DEF_CFG_VISITOR(Binary, Operator, binop) {
    STMT_COUNT(SRCDUMP(binop), binop->getStmtClassName());
    cfg::InternalNode *binop_node = new cfg::InternalNode(m_context, binop);
    m_curr_node->splitEdge(binop_node);
    m_curr_node = binop_node;
    return false;
}

auto ConstructSpmdCFG::get() -> std::vector<cfg::CFGNode *> {
    return m_spmdfy_tutbl;
}

auto ConstructSpmdCFG::add(const clang::VarDecl *var_decl) -> bool {
    m_spmdfy_tutbl.push_back(new cfg::GlobalVarNode(m_context, var_decl));
    return true;
}

auto ConstructSpmdCFG::splitEdge(cfg::CFGNode *node) -> bool {
    SPMDFY_INFO("Adding node {}", node->getNodeTypeName());
    if (m_curr_node == nullptr) {
        SPMDFY_ERROR("Current node is null");
        return true;
    }
    auto next = m_curr_node->getNext();
    if (next == nullptr) {
        SPMDFY_ERROR("Current node is null");
        return true;
    }
    SPMDFY_INFO("Edge splitting from:");
    SPMDFY_INFO("{} -> {}", m_curr_node->getName(), next->getName());
    SPMDFY_INFO("to:");
    node->setNext(next, cfg::CFGEdge::Complete);
    m_curr_node->setNext(node, cfg::CFGEdge::Complete);
    next->setPrevious(node, cfg::CFGEdge::Complete);
    node->setPrevious(m_curr_node, cfg::CFGEdge::Complete);
    m_curr_node = node;
    SPMDFY_INFO("{} -> {} -> {}", m_curr_node->getPrevious()->getName(),
                m_curr_node->getName(), m_curr_node->getNext()->getName());
    return false;
}

auto ConstructSpmdCFG::add(const clang::FunctionDecl *func_decl) -> bool {
    auto func = new cfg::KernelFuncNode(m_context, func_decl);
    m_curr_node = func;
    auto func_exit = new cfg::ExitNode();
    func->setNext(func_exit, cfg::CFGEdge::Complete);
    STMT_COUNT("Entry", "EntryNode");
    TraverseStmt(func_decl->getBody());
    STMT_COUNT("Exit", "ExitNode");
    m_spmdfy_tutbl.push_back(func);
    return false;
}
auto ConstructSpmdCFG::add(const clang::CXXRecordDecl *record_decl) -> bool {
    m_cpp_tutbl.push_back(record_decl);
    return true;
}

} // namespace spmdfy