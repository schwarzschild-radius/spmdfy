#include <spmdfy/Generator/SimpleGenerator.hpp>

namespace spmdfy {

#define DEF_VISITOR(NODE, BASE, NAME)                                          \
    auto SimpleGenerator::Visit##NODE##BASE(const clang::NODE##BASE *NAME)     \
        ->std::string

#define UNUSED(X)
#define DECL_DEF_VISITOR(NODE, NAME) DEF_VISITOR(NODE, Decl, NAME)
#define STMT_DEF_VISITOR(NODE, NAME) DEF_VISITOR(NODE, Stmt, NAME)
#define TYPE_DEF_VISITOR(NODE, NAME) DEF_VISITOR(NODE, Type, NAME)

auto rmCastIf(const clang::Expr *expr) -> const clang::Expr * {
    if (llvm::isa<const clang::ImplicitCastExpr>(expr)) {
        return llvm::cast<const clang::ImplicitCastExpr>(expr)
            ->getSubExprAsWritten();
    }
    return expr;
}

std::string SimpleGenerator::getISPCBaseType(std::string from) {
    std::string to = from;
    if (g_SpmdfyTypeMap.find(from) != g_SpmdfyTypeMap.end()) {
        to = g_SpmdfyTypeMap.at(from);
    }
    SPMDFY_WARN("Converting from {} to {}", from, to);
    return to;
}

DECL_DEF_VISITOR(Declarator, dec_decl) {
    SPMDFY_INFO("Visiting DeclarationDecl: {}", SRCDUMP(dec_decl));
    OStreamTy decl_gen;
    std::string var_name = dec_decl->getNameAsString();
    clang::QualType type = dec_decl->getType();
    clang::PrintingPolicy pm(m_lang_opts);
    std::string var_base_type = type.getAsString(pm);

    if (type.hasQualifiers()) {
        decl_gen << type.getQualifiers().getAsString() << " ";
        var_base_type = type.getUnqualifiedType().getAsString();
    }
    if (g_SpmdfyTypeMap.find(var_base_type) != g_SpmdfyTypeMap.end()) {
        var_base_type = g_SpmdfyTypeMap.at(var_base_type);
    }
    if (!type->isBuiltinType()) {
        decl_gen << var_base_type << "& " << var_name;
    } else {
        decl_gen << var_base_type << " " << var_name;
    }
    return "";
}

DECL_DEF_VISITOR(ParmVar, param_decl) {
    SPMDFY_INFO("Visiting ParmVarDecl: {}", SRCDUMP(param_decl));
    OStreamTy param_gen;
    if (m_tu_context == TUContext::CUDA_KERNEL) {
        clang::QualType param_type = param_decl->getType();
        param_gen << "uniform ";
        if (param_type->isPointerType()) {
            param_gen << getISPCBaseType(
                             param_type->getPointeeType().getAsString())
                      << " ";
            param_gen << param_decl->getNameAsString() << "[]";
        } else {
            param_gen << VisitVarDecl(
                llvm::cast<const clang::VarDecl>(param_decl));
        }
        SPMDFY_WARN("SPMDFIED ParmVarDecl {}", param_gen.str());
    }
    return param_gen.str();
}

auto SimpleGenerator::VisitQualType(clang::QualType qual) -> std::string {
    SPMDFY_INFO("Visiting QualType: {}", qual.getAsString());
    OStreamTy qual_gen;

    if (qual.hasQualifiers()) {
        qual_gen << qual.getQualifiers().getAsString() << " ";
        qual = qual.getUnqualifiedType().getDesugaredType(m_context);
    }

    qual_gen << Visit(qual.getTypePtr());

    return qual_gen.str();
}

TYPE_DEF_VISITOR(Builtin, builtin) {
    clang::PrintingPolicy pm(m_lang_opts);
    SPMDFY_INFO("Visiting BuiltinType: {}", builtin->getName(pm).str());
    OStreamTy type_gen;
    type_gen << getISPCBaseType(builtin->getName(pm));
    return type_gen.str();
}

TYPE_DEF_VISITOR(IncompleteArray, incompl_array) {
    SPMDFY_INFO("Visiting IncompleteArrayType: {}",
                incompl_array->desugar().getAsString());
    OStreamTy incompl_array_gen;
    incompl_array_gen << incompl_array->getElementType().getAsString();
    return incompl_array_gen.str();
}

TYPE_DEF_VISITOR(Pointer, ptr) {
    SPMDFY_INFO("Visiting PointerType:");
    OStreamTy ptr_gen;
    ptr_gen << VisitQualType(ptr->getPointeeType()) << "*";
    return ptr_gen.str();
}

TYPE_DEF_VISITOR(Record, record) {
    SPMDFY_INFO("Visiting RecordType: {}",
                record->getDecl()->getNameAsString());
    OStreamTy record_gen;

    record_gen << record->getDecl()->getNameAsString();

    return record_gen.str();
}

DECL_DEF_VISITOR(Var, var_decl) {
    SPMDFY_INFO("Visiting VarDecl: {}", SRCDUMP(var_decl));
    OStreamTy var_gen;
    std::string var_name = var_decl->getNameAsString();
    clang::QualType type = var_decl->getType();
    std::string var_base_type = VisitQualType(type);

    if (type->isIncompleteType() &&
        var_decl->hasAttr<clang::CUDASharedAttr>()) {
        var_name = "* " + var_name;
        var_name +=
            " = uniform new uniform " + var_base_type + "[shared_memory_size]";
    } else if (type->isConstantArrayType()) {
        do {
            auto const_arr_type = clang::cast<clang::ConstantArrayType>(type);
            var_name =
                var_name + "[" +
                std::to_string((int)*const_arr_type->getSize().getRawData()) +
                "]";
            type = const_arr_type->getElementType();
        } while (type->isConstantArrayType());
        var_base_type = getISPCBaseType(type.getAsString());
    } else if (!type->isBuiltinType() && !type->isPointerType()) {
        SPMDFY_ERROR("Not Builtin Type: {}", type.getAsString());
        var_name = "&" + var_name;
    }

    if (const clang::Expr *initwc = var_decl->getInit();
        (initwc && m_tu_context != TUContext::STRUCT)) {
        const clang::Expr *init = rmCastIf(initwc);
        std::string var_init = SRCDUMP(init);
        if (var_base_type.find("int8") != -1) {
            if (llvm::isa<const clang::CharacterLiteral>(init)) {
                var_init = std::to_string(
                    llvm::cast<const clang::CharacterLiteral>(init)
                        ->getValue());
            }
        }
        var_name += " = ";
        if (type->isPointerType() &&
            llvm::isa<clang::CXXNullPtrLiteralExpr>(init)) {
            var_init = "NULL";
        }
        if (!type->isBuiltinType()) {
            if (llvm::isa<const clang::CXXConstructExpr>(init)) {
                SPMDFY_INFO("Generating CXXConstructExpr");
                const clang::CXXConstructExpr *ctor_expr =
                    llvm::cast<const clang::CXXConstructExpr>(init);
                std::string ctor_type;
                std::vector<std::string> ctor_args;
                for (int i = 0; i < ctor_expr->getNumArgs(); i++) {
                    ctor_type +=
                        "_" + ctor_expr->getArg(i)->getType().getAsString();
                    ctor_args.push_back(SRCDUMP(ctor_expr->getArg(i)));
                }
                var_init = var_base_type + "_ctor" + ctor_type + "(";
                var_init += strJoin(ctor_args.begin(), ctor_args.end());
                var_init += ")";
            }
        }
        var_name += var_init;
    }

    var_gen << var_base_type << " " << var_name;

    if (var_decl->hasAttr<clang::CUDASharedAttr>()) {
        m_shmem << "uniform " << var_gen.str() << ";\n";
        return ";\n";
    }

    if (m_tu_context == TUContext::CUDA_KERNEL && m_scope == 0) {
        m_kernel_context << var_gen.str() << ";\n";
        return var_gen.str();
    }

    if (m_tu_context == TUContext::GLOBAL) {
        var_gen << "uniform ";
        var_gen << var_gen.str();
    }
    return var_gen.str();
}

DEF_VISITOR(Decl, Stmt, decl_stmt) {
    OStreamTy decl_gen;
    SPMDFY_INFO("Visiting DeclStmt: {}", SRCDUMP(decl_stmt));
    for (auto decl : decl_stmt->decls()) {
        decl_gen << VisitVarDecl(llvm::cast<const clang::VarDecl>(decl));
    }
    return decl_gen.str();
}

DEF_VISITOR(Compound, Stmt, cpmd_stmt) {
    OStreamTy cpmd_gen;
    SPMDFY_INFO("Visiting CompoundStmt:");
    if (m_tu_context == TUContext::CUDA_KERNEL) {
        cpmd_gen << "{\n";
        if (m_scope == -1) {
            cpmd_gen << "ISPC_GRID_START" 
                     << "\n"
                     << "ISPC_BLOCK_START"
                     << "\n";
        }
        cpmd_gen << m_kernel_context.str();
        m_scope++;
        for (auto stmt : cpmd_stmt->body()) {
            SPMDFY_WARN("Statment Type: {}", stmt->getStmtClassName());
            if (auto stmt_str = Visit(stmt); stmt_str == std::string()) {
                std::string line = SRCDUMP(stmt);
                cpmd_gen << line << ";\n";
            } else if (stmt_str != ";\n") {
                cpmd_gen << stmt_str;
                if(stmt_str.back() != '}'){
                    cpmd_gen << ";\n";
                }
            }
        }
        m_scope--;
        if (m_scope == -1){
            cpmd_gen << "ISPC_BLOCK_END"
                     << "\n"
                     << "ISPC_GRID_END"
                     << "\n";
        }
        cpmd_gen << "}\n";
    }
    return cpmd_gen.str();
}

DEF_VISITOR(For, Stmt, for_stmt) {
    OStreamTy for_gen;
    SPMDFY_INFO("Visiting ForStmt:");
    auto for_body = for_stmt->getBody();
    if (for_body) {
        m_scope++;
        for_gen << sourceDump(m_sm, m_lang_opts,
                              for_stmt->getSourceRange().getBegin(),
                              for_body->getSourceRange().getBegin());
        for_gen << Visit(for_body);
        m_scope--;
    }
    return for_gen.str();
}

DEF_VISITOR(If, Stmt, if_stmt) {
    OStreamTy if_gen;
    SPMDFY_INFO("Visiting IfStmt:");
    if_gen << "if (";
    const clang::Expr *if_cond = if_stmt->getCond();
    if (if_cond) {
        if_gen << sourceDump(m_sm, m_lang_opts,
                             if_cond->getSourceRange().getBegin(),
                             if_cond->getSourceRange().getEnd())
               << ")";
    }
    const clang::Stmt *if_then = if_stmt->getThen();
    if (if_then) {
        m_scope++;
        if_gen << Visit(if_then);
    }
    m_scope--;
    const clang::Stmt *if_else = if_stmt->getElse();
    if (if_else) {
        if_gen << " else ";
        if (llvm::isa<clang::IfStmt>(if_else)) {
            if_gen << Visit(if_else);
            return if_gen.str();
        }
        m_scope++;
        if_gen << Visit(if_else);
        m_scope--;
    }
    return if_gen.str();
}

auto SimpleGenerator::VisitBinaryOperator(const clang::BinaryOperator *binop)
    -> std::string {
    SPMDFY_INFO("Visiting BinaryOperatorExpr: {}", SRCDUMP(binop));
    OStreamTy binop_gen;
    binop_gen << SRCDUMP(binop);
    return binop_gen.str();
}

DEF_VISITOR(Call, Expr, call_expr) {
    SPMDFY_INFO("Visiting CallExpr: {}", SRCDUMP(call_expr));
    OStreamTy call_gen;
    const clang::FunctionDecl *callee = call_expr->getDirectCallee();
    std::string callee_name = callee->getNameAsString();
    if (callee_name == "__syncthreads") {
        call_gen << "SYNCTHREADS()\n";
        call_gen << m_kernel_context.str();
        return call_gen.str();
    } else if (callee_name == "printf") {
        return "";
    }
    if (auto is_atomic = g_SpmdfyAtomicMap.find(callee_name);
        is_atomic != g_SpmdfyAtomicMap.end()) {
        const auto args = call_expr->getArgs();
        call_gen << is_atomic->second << "(";
        if (is_atomic->first != "atomicCAS") {
            call_gen << SRCDUMP(args[0]) << ", " << SRCDUMP(args[1]);
        } else {
            call_gen << SRCDUMP(args[0]) << ", " << SRCDUMP(args[1]) << ", "
                     << SRCDUMP(args[2]);
        }
        call_gen << ")";
        return call_gen.str();
    }
    call_gen << SRCDUMP(call_expr);
    return call_gen.str();
}

DECL_DEF_VISITOR(Function, func_decl) {
    SPMDFY_INFO("Visiting FunctionDecl {}", func_decl->getNameAsString());
    OStreamTy func_gen;
    if (func_decl->hasAttr<clang::CUDAGlobalAttr>()) {
        m_tu_context = TUContext::CUDA_KERNEL;
        std::string func_body = Visit(func_decl->getBody());
        func_gen << "ISPC_KERNEL(" << func_decl->getNameAsString();
        auto params = func_decl->parameters();
        for (auto param : params) {
            func_gen << ", " << Visit(param);
        }
        func_gen << "){\n";
        func_gen << "ISPC_GRID_START"
                 << "\n";
        func_gen << m_shmem.str();
        func_gen << func_body;
        func_gen << "ISPC_GRID_END"
                 << "\n";
        func_gen << "}\n";
    } else if (func_decl->hasAttr<clang::CUDADeviceAttr>()) {
        m_tu_context = TUContext::CUDA_DEVICE_FUNCTION;

        func_gen << "ISPC_DEVICE_FUNCTION(" << func_decl->getNameAsString();
        func_gen << "){\n";
        func_gen << "}\n";
        m_tu_context = TUContext::GLOBAL;
    }
    return func_gen.str();
}

DECL_DEF_VISITOR(Field, field) {
    SPMDFY_INFO("Visiting Field: {}", SRCDUMP(field));
    OStreamTy field_gen;
    clang::QualType type = field->getType();
    field_gen << VisitQualType(type) << " " << field->getNameAsString();
    return field_gen.str();
}

DECL_DEF_VISITOR(CXXConstructor, ctor) {
    SPMDFY_INFO("Visiting CXXConstructorDecl: {}", SRCDUMP(ctor));
    OStreamTy ctor_gen;

    return ctor_gen.str();
}

DECL_DEF_VISITOR(CXXRecord, struct_decl) {
    if (!struct_decl->isStruct() || m_tu_context != TUContext::GLOBAL ||
        std::none_of(
            struct_decl->ctor_begin(), struct_decl->ctor_end(),
            [](auto ctor_decl) {
                if (ctor_decl->isDefaultConstructor() &&
                    ctor_decl->template hasAttr<clang::CUDADeviceAttr>()) {
                    return true;
                }
                return false;
            })) {
        SPMDFY_ERROR("Cannot spmdfy {} struct", struct_decl->getNameAsString());
        return "";
    }

    SPMDFY_INFO("Visiting CXXRecordDecl: {}", SRCDUMP(struct_decl));
    OStreamTy struct_gen;
    m_tu_context = TUContext::STRUCT;

    std::string struct_name = struct_decl->getNameAsString();
    struct_gen << "// Struct\n";
    struct_gen << "struct " << struct_name << "{\n";
    // members
    struct_gen << "// Fields\n";
    for (auto field : struct_decl->fields()) {
        struct_gen << Visit(field) << ";\n";
    }
    struct_gen << "};\n";

    // generate constructors

    m_tu_context = TUContext::GLOBAL;
    return struct_gen.str();
}

DECL_DEF_VISITOR(Namespace, ns_decl) { return ""; }

} // namespace spmdfy