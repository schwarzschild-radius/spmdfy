#include <spmdfy/Generator/CFGGenerator/ConstructCFG.hpp>

namespace spmdfy {

#define DEF_CFG_VISITOR(NODE, BASE, PARAM)                                     \
    auto ConstructSpmdCFG::Visit##NODE##BASE(clang::NODE##BASE *PARAM)->bool

#define STMT_COUNT(STMT)                                                       \
    SPMDFY_INFO("s{} {}", m_stmt_count, STMT);                                 \
    m_stmt_count++;

DEF_CFG_VISITOR(Compound, Stmt, cpmd) {
    for (auto stmt : cpmd->body()) {
        if (!TraverseStmt(stmt))
            continue;
        STMT_COUNT(stmt->getStmtClassName());
    }
    return false;
}

DEF_CFG_VISITOR(Decl, Stmt, decl) {
    STMT_COUNT(SRCDUMP(decl));

    return false;
}

DEF_CFG_VISITOR(For, Stmt, for_stmt) {
    STMT_COUNT(sourceDump(m_sm, m_lang_opts, for_stmt->getForLoc(),
                          for_stmt->getRParenLoc()));

    TraverseStmt(for_stmt->getBody());
    STMT_COUNT("Reconv }");

    return false;
}

DEF_CFG_VISITOR(If, Stmt, if_stmt) {
    STMT_COUNT(sourceDump(m_sm, m_lang_opts, if_stmt->getBeginLoc(),
                          if_stmt->getCond()->getEndLoc()));

    if (if_stmt->getThen())
        TraverseStmt(if_stmt->getThen());
    if (if_stmt->getElse())
        TraverseStmt(if_stmt->getElse());
    STMT_COUNT("Reconv }");
    return false;
}

DEF_CFG_VISITOR(Call, Expr, call) {
    STMT_COUNT(SRCDUMP(call));
    return false;
}

DEF_CFG_VISITOR(PseudoObject, Expr, pseudo) { return false; }

DEF_CFG_VISITOR(CompoundAssign, Operator, assgn) {
    STMT_COUNT(SRCDUMP(assgn));
    return false;
}

DEF_CFG_VISITOR(Binary, Operator, binop) {
    STMT_COUNT(SRCDUMP(binop));
    return false;
}

auto ConstructSpmdCFG::get() -> std::shared_ptr<CFGNode> {
    STMT_COUNT("Entry");
    TraverseStmt(m_cpmd_stmt);
    STMT_COUNT("Exit");
    return nullptr;
}

} // namespace spmdfy