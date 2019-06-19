#include <spmdfy/SpmdfyStmtVisitor.hpp>

namespace spmdfy {
clang::Stmt *
SpmdfyStmtVisitor::VisitCompoundStmt(clang::CompoundStmt *cpmd_stmt) {
    llvm::errs() << "Visiting Compound Statment\n";
    for (auto stmt = cpmd_stmt->body_begin(); stmt != cpmd_stmt->body_end();
         stmt++) {
        if (Visit(*stmt))
            continue;
        std::string line = sourceDump(m_sm, m_lang_opt, *stmt);
        llvm::errs() << line << '\n';
        m_function_body[m_block].push_back(line);
    }
    if (m_function_body[m_block].size() != 0)
        m_function_body[m_block].back() =
            (std::string)m_function_body[m_block].back() + ";";
    return cpmd_stmt;
}

clang::Stmt *SpmdfyStmtVisitor::VisitCallExpr(clang::CallExpr *call_expr) {
    llvm::errs() << "Visiting Call Expr\n";
    clang::FunctionDecl *callee = call_expr->getDirectCallee();
    std::string callee_name = callee->getNameAsString();
    if (callee_name == "__syncthreads") {
        m_block++;
        return call_expr;
    }else if(callee_name == "printf"){
        return call_expr;
    }
    std::unordered_map<std::string_view, std::string_view> atomic_map = {
        {"atomicAdd", "atomic_add_global"},
        {"atomicSub", "atomic_subtract_global"},
        {"atomicExch", "atomic_swap_global"},
        {"atomicMin", "atomic_min_global"},
        {"atomicMax", "atomic_max_global"},
        {"atomicCAS", "atomic_compare_exchange"}};
    auto is_atomic = atomic_map.find(callee_name);
    if (is_atomic != atomic_map.end()) {
        std::ostringstream ss;
        clang::Expr **args = call_expr->getArgs();
        ss << is_atomic->second << "(";
        if (is_atomic->first != "atomicCAS") {
            ss << sourceDump(m_sm, m_lang_opt, args[0]) << ", "
               << sourceDump(m_sm, m_lang_opt, args[1]);
        } else {
            ss << sourceDump(m_sm, m_lang_opt, args[0]) << ", "
               << sourceDump(m_sm, m_lang_opt, args[1]) << ", "
               << sourceDump(m_sm, m_lang_opt, args[2]);
        }
        ss << ");\n";
        m_function_body[m_block].push_back(ss.str());
        return call_expr;
    }
    m_function_body[m_block].push_back(sourceDump(m_sm, m_lang_opt, call_expr));
    return call_expr;
}

clang::Stmt *SpmdfyStmtVisitor::VisitDeclStmt(clang::DeclStmt *decl_stmt) {
    llvm::errs() << "Visiting Decl Statement\n";
    for (auto dgr : decl_stmt->getDeclGroup()) {
        if (llvm::isa<clang::VarDecl>(dgr)) {
            auto var_decl = llvm::cast<clang::VarDecl>(dgr);
            if (var_decl->hasAttr<clang::CUDASharedAttr>()) { // parsing shared
                                                              // memory
                clang::QualType arr_type = var_decl->getType();
                llvm::errs() << sourceDump(m_sm, m_lang_opt,
                                           var_decl->getTypeSpecStartLoc(),
                                           var_decl->getSourceRange().getEnd())
                             << '\n';

                m_shmem.push_back(sourceDump(
                    m_sm, m_lang_opt, var_decl->getTypeSpecStartLoc(),
                    var_decl->getSourceRange().getEnd()));
                return decl_stmt;
            }
        }
    }
    m_function_body[m_block].push_back(sourceDump(m_sm, m_lang_opt, decl_stmt));
    return decl_stmt;
}

clang::Stmt *SpmdfyStmtVisitor::VisitForStmt(clang::ForStmt *for_stmt) {
    llvm::errs() << "Visiting For Statement\n";
    auto for_body = for_stmt->getBody();
    if (for_body) {
        m_function_body[m_block].push_back(
            sourceDump(m_sm, m_lang_opt, for_stmt->getSourceRange().getBegin(),
                       for_body->getSourceRange().getBegin()));
        Visit(for_body);
        m_function_body[m_block].push_back("}\n");
    }
    return for_stmt;
}

} // namespace spmdfy