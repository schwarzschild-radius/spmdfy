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
    } else if (callee_name == "printf") {
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

clang::QualType getBaseType(clang::DeclaratorDecl *decl) {
    clang::TypeSourceInfo *type_info = decl->getTypeSourceInfo();
    clang::QualType type = type_info->getType();
    return type.getCanonicalType();
}

bool hasIncompleteType(clang::DeclaratorDecl *decl) {
    clang::TypeSourceInfo *type_info = decl->getTypeSourceInfo();
    clang::QualType type = type_info->getType();
    return type->isIncompleteType();
}

clang::Stmt *SpmdfyStmtVisitor::VisitDeclStmt(clang::DeclStmt *decl_stmt) {
    llvm::errs() << "Visiting Decl Statement\n";
    for (auto dgr : decl_stmt->getDeclGroup()) {
        if (llvm::isa<clang::VarDecl>(dgr)) {
            auto var_decl = llvm::cast<clang::VarDecl>(dgr);
            std::string var_name = var_decl->getNameAsString();
            clang::QualType var_type = var_decl->getTypeSourceInfo()->getType();
            std::string type_str = "";
            if (var_decl->hasAttr<clang::CUDASharedAttr>()) { // parsing shared
                                                              // memory
                if (var_decl->hasExternalStorage()) {
                    auto &extern_shmem = m_extern_shmem[var_name];
                    if (hasIncompleteType(var_decl)) {
                        extern_shmem["type"] = var_type->getAsArrayTypeUnsafe()
                                                   ->getElementType()
                                                   .getAsString();
                        extern_shmem["type_kind"] = "IncompleteType";
                        return decl_stmt;
                    }
                    extern_shmem["type"] = var_type.getAsString();
                    extern_shmem["type_kind"] = "Built-in";
                    return decl_stmt;
                }
                auto &shmem = m_shmem[var_name];
                if (var_type->isConstantArrayType()) {
                    int count = 0;
                    shmem["array_dim_value"] = {};
                    do {
                        auto const_arr_type =
                            clang::cast<clang::ConstantArrayType>(var_type);
                        shmem["array_dim_value"].push_back(
                            (int)*const_arr_type->getSize().getRawData());
                        count++;
                        var_type = const_arr_type->getElementType();
                    } while (var_type->isConstantArrayType());
                    shmem["array_dims"] = count;
                    shmem["type_kind"] = "array_type";
                    shmem["type"] = var_type.getAsString();
                    return decl_stmt;
                }
                shmem["type"] = var_type.getAsString();
                shmem["type_kind"] = "Built-in";
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