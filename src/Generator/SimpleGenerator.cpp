#include <spmdfy/Generator/SimpleGenerator.hpp>

namespace spmdfy {

std::string ispc_macros = R"macro(
#define ISPC_GRID_START                                                        \
    Dim3 blockIdx, threadIdx;                                                  \
    for (blockIdx.z = 0; blockIdx.z < gridDim.z; blockIdx.z++) {               \
        for (blockIdx.y = 0; blockIdx.y < gridDim.y; blockIdx.y++) {           \
            for (blockIdx.x = 0; blockIdx.x < gridDim.x; blockIdx.x++) {

#define ISPC_BLOCK_START                                                       \
    for (threadIdx.z = 0; threadIdx.z < blockDim.z; threadIdx.z++) {           \
        for (threadIdx.y = 0; threadIdx.y < blockDim.y; threadIdx.y++) {       \
            for (threadIdx.x = programIndex; threadIdx.x < blockDim.x;         \
                 threadIdx.x += programCount) {

#define ISPC_GRID_END                                                          \
    }                                                                          \
    }                                                                          \
    }

#define ISPC_BLOCK_END                                                         \
    }                                                                          \
    }                                                                          \
    }

#define ISPC_START                                                             \
    ISPC_GRID_START                                                            \
    ISPC_BLOCK_START

#define ISPC_END                                                               \
    ISPC_GRID_END                                                              \
    ISPC_BLOCK_END

#define SYNCTHREADS()                                                          \
    ISPC_BLOCK_END                                                             \
    ISPC_BLOCK_START

#define ISPC_KERNEL(function, ...)                                             \
    export void function(                                                      \
        const uniform Dim3 &gridDim, const uniform Dim3 &blockDim,             \
        const uniform size_t &shared_memory_size, __VA_ARGS__)

#define ISPC_DEVICE_FUNCTION(rety, function, ...)                              \
    rety function(const uniform Dim3 &gridDim, const uniform Dim3 &blockDim,   \
                  const Dim3 &blockIdx, const Dim3 &threadIdx, __VA_ARGS__)

#define ISPC_DEVICE_CALL(function, ...)                                        \
    function(gridDim, blockDim, blockIdx, threadIdx, __VA_ARGS__)

#define NS(x, y) x##_##y
#define NS3(x, y, z) x##_##y##_##z
#define ENUM(x, y) const int x##_##y
struct Dim3 {
    int x, y, z;
};

)macro";

#define DEF_VISITOR(NODE, BASE, NAME)                                          \
    auto SimpleGenerator::Visit##NODE##BASE(const clang::NODE##BASE *NAME)     \
        ->std::string

#define DECL_DEF_VISITOR(NODE, NAME) DEF_VISITOR(NODE, Decl, NAME)

#define SRCDUMP(NODE) sourceDump(m_sm, m_lang_opts, NODE)

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
    std::ostringstream decl_gen;
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
    std::ostringstream param_gen;
    if (m_tu_context == TUContext::CUDA_KERNEL) {
        clang::QualType param_type = param_decl->getType();
        param_gen << "uniform ";
        if(param_type->isPointerType()){
            param_gen << getISPCBaseType(param_type->getPointeeType().getAsString()) << " ";
            param_gen << param_decl->getNameAsString() << "[]";
        }else{
            param_gen << VisitVarDecl(llvm::cast<const clang::VarDecl>(param_decl));
        }
        SPMDFY_WARN("SPMDFIED ParmVarDecl {}", param_gen.str());
    }
    return param_gen.str();
}

DECL_DEF_VISITOR(Var, var_decl) {
    SPMDFY_INFO("Visiting VarDecl: {}", SRCDUMP(var_decl));
    std::ostringstream var_gen;
    std::string var_name = var_decl->getNameAsString();
    clang::QualType type = var_decl->getType().getDesugaredType(m_context);
    clang::PrintingPolicy pm(m_lang_opts);
    std::string var_base_type = getISPCBaseType(type.getAsString(pm));

    if (type.hasQualifiers()) {
        var_gen << type.getQualifiers().getAsString() << " ";
        var_base_type = type.getUnqualifiedType().getDesugaredType(m_context).getAsString();
        var_base_type = getISPCBaseType(var_base_type);
        SPMDFY_WARN("VarType: {}", var_base_type);
    }

    if (type->isIncompleteType()) {
        var_base_type =
            type->getAsArrayTypeUnsafe()->getElementType().getAsString();
        var_base_type = getISPCBaseType(var_base_type);
        var_name = "* " + var_name;
        var_name += " = uniform new uniform " + getISPCBaseType(var_base_type) +
                    "[shared_memory_size]";
    } else if (type->isConstantArrayType()) {
        do {
            auto const_arr_type = clang::cast<clang::ConstantArrayType>(type);
            var_name =
                var_name + "[" +
                std::to_string((int)*const_arr_type->getSize().getRawData()) +
                "]";
            type = const_arr_type->getElementType();
        } while (type->isConstantArrayType());
        var_base_type = type.getAsString();
    } else if (!type->isBuiltinType() && !type->isPointerType()) {
        SPMDFY_ERROR("Not Builtin Type: {}", type.getAsString());
        var_name = "&" + var_name;
    }

    if (const clang::Expr *init = var_decl->getInit();
        init && m_tu_context != TUContext::STRUCT) {
        std::string var_init = SRCDUMP(init);
        if (var_base_type.find("int8") != -1) {
            if (llvm::isa<const clang::CharacterLiteral>(rmCastIf(init))) {
                var_init = std::to_string(
                    llvm::cast<const clang::CharacterLiteral>(rmCastIf(init))
                        ->getValue());
            }
        }
        var_name += " = ";
        if (!type->isBuiltinType()) {
            if (llvm::isa<const clang::CXXConstructExpr>(init)) {
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
        return "";
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
    std::ostringstream decl_gen;
    SPMDFY_INFO("Visiting DeclStmt: {}", SRCDUMP(decl_stmt));
    for (auto decl : decl_stmt->decls()) {
        decl_gen << VisitVarDecl(llvm::cast<const clang::VarDecl>(decl));
    }
    return decl_gen.str();
}

DEF_VISITOR(Compound, Stmt, cpmd_stmt) {
    std::ostringstream cpmd_gen;
    SPMDFY_INFO("Visiting CompoundStmt: {}", SRCDUMP(cpmd_stmt));
    if (m_tu_context == TUContext::CUDA_KERNEL) {
        cpmd_gen << "ISPC_BLOCK_START"
                 << "\n";
        cpmd_gen << m_kernel_context.str();
        m_scope++;
        for (auto stmt = cpmd_stmt->body_begin(); stmt != cpmd_stmt->body_end();
             stmt++) {
            SPMDFY_WARN("Statment Type: {}", (*stmt)->getStmtClassName());
            if (auto stmt_str = Visit(*stmt); stmt_str != "") {
                cpmd_gen << stmt_str << ";\n";
                continue;
            }
            std::string line = SRCDUMP(*stmt);
            if (line.back() != ';' && line.size())
                cpmd_gen << line << ";\n";
        }
        cpmd_gen << "ISPC_BLOCK_END"
                 << "\n";
        m_scope--;
    }
    return cpmd_gen.str();
}

DEF_VISITOR(For, Stmt, for_stmt) {
    std::ostringstream for_gen;
    SPMDFY_INFO("Visiting ForStmt: {}", SRCDUMP(for_stmt));
    auto for_body = for_stmt->getBody();
    if (for_body) {
        m_scope++;
        for_gen << sourceDump(m_sm, m_lang_opts,
                              for_stmt->getSourceRange().getBegin(),
                              for_body->getSourceRange().getBegin());
        for_gen << Visit(for_body);
        m_scope--;
        for_gen << "}\n";
    }
    return for_gen.str();
}

DEF_VISITOR(If, Stmt, if_stmt) {
    std::ostringstream if_gen;
    SPMDFY_INFO("Visiting IfStmt: {}", SRCDUMP(if_stmt));
    if_gen << "if (";
    const clang::Expr *if_cond = if_stmt->getCond();
    if (if_cond) {
        if_gen << sourceDump(m_sm, m_lang_opts,
                             if_cond->getSourceRange().getBegin(),
                             if_cond->getSourceRange().getEnd())
               << ")";
    }
    if_gen << "{\n";
    const clang::Stmt *if_then = if_stmt->getThen();
    if (if_then) {
        m_scope++;
        if_gen << Visit(if_then);
    }
    if_gen << "}";
    m_scope--;
    const clang::Stmt *if_else = if_stmt->getElse();
    if (if_else) {
        if_gen << " else ";
        if (llvm::isa<clang::IfStmt>(if_else)) {
            if_gen << Visit(if_else);
            return if_gen.str();
        }
        if_gen << "{\n";
        m_scope++;
        if_gen << Visit(if_else);
        if_gen << "}";
        m_scope--;
    }
    if_gen << "\n";
    return if_gen.str();
}

auto SimpleGenerator::VisitBinaryOperator(const clang::BinaryOperator *binop)
    -> std::string {
    SPMDFY_INFO("Visiting BinaryOperatorExpr: {}", SRCDUMP(binop));
    std::ostringstream binop_gen;
    binop_gen << SRCDUMP(binop) << ";\n";
    return binop_gen.str();
}

DEF_VISITOR(Call, Expr, call_expr) {
    SPMDFY_INFO("Visiting CallExpr: {}", SRCDUMP(call_expr));
    std::ostringstream call_gen;
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
    std::ostringstream func_gen;
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

DECL_DEF_VISITOR(CXXRecord, struct_decl) {
    if (m_tu_context != TUContext::GLOBAL) {
        return "";
    }
    SPMDFY_INFO("Visiting CXXRecordDecl: {}", SRCDUMP(struct_decl));
    std::ostringstream struct_gen;
    m_tu_context = TUContext::STRUCT;

    std::string struct_name = struct_decl->getNameAsString();
    struct_gen << "// Struct\n";
    struct_gen << "struct " << struct_name << "{\n";
    // members
    struct_gen << "// Fields\n";
    for (auto field = struct_decl->field_begin();
         field != struct_decl->field_end(); field++) {
        struct_gen << Visit(llvm::cast<const clang::DeclaratorDecl>(*field));
        struct_gen << ";\n";
    }
    struct_gen << "};\n";
    m_tu_context = TUContext::GLOBAL;
    return struct_gen.str();
}

DECL_DEF_VISITOR(Namespace, ns_decl) { return ""; }

} // namespace spmdfy