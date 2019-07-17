#include <spmdfy/SpmdfyStmtVisitor.hpp>

namespace spmdfy {
clang::Stmt *
SpmdfyStmtVisitor::VisitCompoundStmt(clang::CompoundStmt *cpmd_stmt) {
    m_scope++;
    llvm::errs() << "Visiting Compound Statment\n";
    for (auto stmt = cpmd_stmt->body_begin(); stmt != cpmd_stmt->body_end();
         stmt++) {
        if (Visit(*stmt))
            continue;
        llvm::errs() << "Statment: ";
        std::string line = sourceDump(m_sm, m_lang_opt, *stmt);
        if (line.back() != ';' && line.size() != 0 && line != "\n")
            line += ';';
        llvm::errs() << line << '\n';
        m_function_body[m_block].push_back(line);
    }
    m_scope--;
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
            nl::json var_metadata;
            std::string var_name = var_decl->getNameAsString();
            var_metadata["name"] = var_name;
            clang::QualType var_type = var_decl->getTypeSourceInfo()->getType();
            var_metadata["type"]["base_type"] = var_type.getAsString();
            var_metadata["type"]["type_kind"] = "Built-in";
            if (var_type.hasQualifiers())
                var_metadata["type"]["qualifiers"] =
                    var_type.getQualifiers().getAsString();
            clang::Expr *var_init = var_decl->getInit();
            if (var_init) {
                var_metadata["init"] =
                    sourceDump(m_sm, m_lang_opt, var_init);
            }
            if (var_type->isIncompleteType()) {
                var_metadata["type"]["base_type"] = var_type->getAsArrayTypeUnsafe()
                                                    ->getElementType()
                                                    .getAsString();
                var_metadata["type"]["type_kind"] = "IncompleteType";
            }else if(var_type->isConstantArrayType()) {
                int count = 0;
                var_metadata["type"]["array_dim_value"] = {};
                do {
                    auto const_arr_type =
                        clang::cast<clang::ConstantArrayType>(var_type);
                    var_metadata["type"]["array_dim_value"].push_back(
                        (int)*const_arr_type->getSize().getRawData());
                    count++;
                    var_type = const_arr_type->getElementType();
                } while (var_type->isConstantArrayType());
                var_metadata["type"]["array_dims"] = count;
                var_metadata["type"]["type_kind"] = "array_type";
                var_metadata["type"]["base_type"] = var_type.getAsString();
            }
            if (var_decl->hasAttr<clang::CUDASharedAttr>()) { // parsing shared
                                                              // memory
                if (var_decl->hasExternalStorage()) {
                    m_extern_shmem.push_back(var_metadata);
                    return decl_stmt;
                }
                m_shmem.push_back(var_metadata);
            }
            llvm::errs() << sourceDump(m_sm, m_lang_opt, decl_stmt) << '\n';
            if (m_scope == 0) {
                m_context.push_back(var_metadata);
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

clang::Stmt *SpmdfyStmtVisitor::VisitIfStmt(clang::IfStmt *if_stmt) {
    llvm::errs() << "Visiting If Statement\n";
    std::ostringstream if_stmt_str;
    m_function_body[m_block].push_back("if (");
    clang::Stmt *if_init = if_stmt->getInit();
    if (if_init) {
        llvm::errs() << "ifInit: "
                     << sourceDump(m_sm, m_lang_opt,
                                   if_init->getSourceRange().getBegin(),
                                   if_init->getSourceRange().getEnd())
                     << ";\n";
        m_function_body[m_block].push_back(
            sourceDump(m_sm, m_lang_opt, if_init->getSourceRange().getBegin(),
                       if_init->getSourceRange().getEnd()) +
            ";");
    }
    clang::Expr *if_cond = if_stmt->getCond();
    if (if_cond) {
        llvm::errs() << sourceDump(m_sm, m_lang_opt,
                                   if_cond->getSourceRange().getBegin(),
                                   if_cond->getSourceRange().getEnd());
        m_function_body[m_block].push_back(
            sourceDump(m_sm, m_lang_opt, if_cond->getSourceRange().getBegin(),
                       if_cond->getSourceRange().getEnd()) +
            ")");
    }
    m_function_body[m_block].push_back("{\n");
    clang::Stmt *if_then = if_stmt->getThen();
    if (if_then) {
        llvm::errs() << "ifThen:\n";
        /*         m_function_body[m_block].push_back(sourceDump(m_sm,
           m_lang_opt, if_then->getSourceRange().getBegin(),
                                          if_then->getSourceRange().getBegin()));
         */
        Visit(if_then);
    }
    m_function_body[m_block].push_back("}");
    clang::Stmt *if_else = if_stmt->getElse();
    if (if_else) {
        m_function_body[m_block].push_back(" else ");
        llvm::errs() << "ifElse:\n";
        if (llvm::isa<clang::IfStmt>(if_else)) {
            Visit(if_else);
            return if_else;
        }
        m_function_body[m_block].push_back("{\n");
        Visit(if_else);
        m_function_body[m_block].push_back("}\n");
    }
    return if_stmt;
}

} // namespace spmdfy