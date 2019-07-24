#include <spmdfy/SpmdfyAction.hpp>

namespace spmdfy {

std::unique_ptr<clang::ASTConsumer> SpmdfyAction::newASTConsumer() {
    m_finder.reset(new clang::ast_matchers::MatchFinder);
    // match kernel function
    m_finder->addMatcher(
        mat::functionDecl(mat::hasAttr(clang::attr::CUDAGlobal),
                          mat::isDefinition())
            .bind("cudaKernelFunction"),
        this);
    // match device function
    m_finder->addMatcher(
        mat::functionDecl(mat::hasAttr(clang::attr::CUDADevice),
                          mat::isExpansionInMainFile(), mat::isDefinition(),
                          mat::unless(mat::cxxMethodDecl()))
            .bind("cudaDeviceFunction"),
        this);
    // match global decls
    m_finder->addMatcher(
        mat::varDecl(mat::isExpansionInMainFile(), mat::hasGlobalStorage(),
                     mat::hasParent(mat::translationUnitDecl()))
            .bind("globalDeclarations"),
        this);
    // match struct types
    m_finder->addMatcher(
        mat::cxxRecordDecl(mat::isStruct(), mat::isExpansionInMainFile(),
                           mat::has(mat::cxxConstructorDecl(
                               mat::hasAttr(clang::attr::CUDADevice))))
            .bind("structType"),
        this);
    m_finder->addMatcher(
        mat::enumDecl(mat::isExpansionInMainFile()).bind("EnumType"), this);
    return m_finder->newASTConsumer();
}

std::unique_ptr<clang::ASTConsumer>
SpmdfyAction::CreateASTConsumer(clang::CompilerInstance &, llvm::StringRef) {
    return newASTConsumer();
}

bool SpmdfyAction::cudaKernelFunction(
    const mat::MatchFinder::MatchResult &result) {
    llvm::StringRef ref("cudaKernelFunction");
    auto *kernel_function = result.Nodes.getNodeAs<clang::FunctionDecl>(ref);
    if (!kernel_function || !kernel_function->doesThisDeclarationHaveABody()) {
        return false;
    }
    nl::json metadata = {};
    llvm::errs() << ref.str() << '\n';
    llvm::errs() << kernel_function->getNameAsString() << "\n";
    // 1. Name
    std::string name = kernel_function->getNameAsString();

    clang::SourceManager &sm = *result.SourceManager;
    clang::LangOptions lang_opt;
    // 2. Export
    metadata["exported"] = true;

    // 3. Params
    metadata["params"] = {};
    for (size_t param_idx = 0; param_idx < kernel_function->getNumParams();
         param_idx++) {
        auto param = kernel_function->getParamDecl(param_idx);
        clang::QualType param_type = param->getOriginalType();
        llvm::errs() << param_type.getAsString() << ' '
                     << param_type->isPointerType() << '\n';
        if (param_type->isPointerType()) {
            std::string pointee_type =
                param_type->getPointeeType().getAsString();
            metadata["params"].push_back(pointee_type + " " +
                                         param->getNameAsString() + "[]");
        } else {
            metadata["params"].push_back(sourceDump(sm, lang_opt, param));
        }
    }
    // 4. body
    clang::Stmt *body = kernel_function->getBody();
    if (body) {
        m_stmt_visitor->Visit(body);
    }
    metadata["context"] = m_stmt_visitor->getContext();
    metadata["body"] = m_stmt_visitor->getFunctionBody();
    metadata["shmem"] = m_stmt_visitor->getSharedMem();
    metadata["extern_shmem"] = m_stmt_visitor->getExternSharedMem();
    m_function_metadata["functions"][name] = metadata;
    return true;
}

bool SpmdfyAction::cudaDeviceFunction(
    const mat::MatchFinder::MatchResult &result) {
    llvm::StringRef ref("cudaDeviceFunction");
    auto *device_function = result.Nodes.getNodeAs<clang::FunctionDecl>(ref);
    if (!device_function || !device_function->doesThisDeclarationHaveABody()) {
        return false;
    }
    nl::json metadata = {};
    llvm::errs() << ref.str() << '\n';
    llvm::errs() << device_function->getNameAsString() << "\n";
    clang::SourceManager &sm = *result.SourceManager;
    clang::LangOptions lang_opt;
    // 1. Name
    std::string name = device_function->getNameAsString();

    // 2. Export
    metadata["exported"] = false;

    // 3. Return Type
    std::string return_type = device_function->getReturnType().getAsString();
    metadata["return_type"] = return_type;

    // 4. Params
    metadata["params"] = {};
    for (auto param_idx = 0; param_idx < device_function->getNumParams();
         param_idx++) {
        auto param = device_function->getParamDecl(param_idx);
        metadata["params"].push_back(sourceDump(sm, lang_opt, param));
    }
    clang::Stmt *body = device_function->getBody();
    if (body) {
        m_stmt_visitor->Visit(body);
    }
    metadata["body"] = m_stmt_visitor->getFunctionBody();
    m_function_metadata["functions"][name] = metadata;
    return true;
}

bool SpmdfyAction::globalDeclarations(
    const mat::MatchFinder::MatchResult &result) {
    llvm::StringRef ref("globalDeclarations");
    auto *var_decl = result.Nodes.getNodeAs<clang::VarDecl>(ref);
    if (!var_decl) {
        return false;
    }
    llvm::errs() << ref.str() << '\n';
    clang::SourceManager &sm = *result.SourceManager;
    clang::LangOptions lang_opt = result.Context->getLangOpts();
    for(auto i : lang_opt.ModuleFeatures){
        llvm::errs() << i << '\n';
    }
    if (var_decl->hasAttr<clang::CUDAConstantAttr>()) {
        llvm::errs() << sourceDump(sm, lang_opt,
                                   var_decl->getTypeSpecStartLoc(),
                                   var_decl->getSourceRange().getEnd())
                     << '\n';
        m_function_metadata["globals"].push_back(
            sourceDump(sm, lang_opt, var_decl->getTypeSpecStartLoc(),
                       var_decl->getSourceRange().getEnd()) +
            ";");
    } else {
        nl::json var_metadata;
        var_metadata["name"] = var_decl->getNameAsString();
        var_decl->dump();
        clang::QualType var_type = var_decl->getType();
        var_type->dump();
        std::string base_type = var_type.getCanonicalType().getAsString();
/*         if(base_type == "_Bool"){
            base_type = "bool";
        }else if (base_type == "char"){
            base_type = "int8";
        } */
        if(pm.Bool == 1){
            llvm::errs() << "We are in c land?\n";
        }
        var_metadata["type"]["base_type"] = base_type;
        var_metadata["type"]["type_kind"] = "Built-in";
        if (var_type.hasQualifiers())
            var_metadata["type"]["qualifiers"] =
                var_type.getQualifiers().getAsString();
        const clang::Expr *var_init = var_decl->getInit();
        if (var_init) {
            var_metadata["init"] = sourceDump(sm, lang_opt, var_init);
        }
        m_function_metadata["globals"].push_back(var_metadata);
    }
    return true;
}

bool SpmdfyAction::structType(const mat::MatchFinder::MatchResult &result) {
    llvm::StringRef ref("structType");
    auto *struct_type = result.Nodes.getNodeAs<clang::CXXRecordDecl>(ref);
    if (!struct_type) {
        return false;
    }
    clang::SourceManager &sm = *result.SourceManager;
    clang::LangOptions lang_opt;
    llvm::errs() << "StructType\n";
    llvm::errs() << struct_type->getNameAsString() << '\n';
    nl::json metadata;
    metadata["name"] = struct_type->getNameAsString();
    llvm::errs() << sourceDump(sm, lang_opt, struct_type) << '\n';
    // members
    llvm::errs() << "Fields\n";
    for (auto field = struct_type->field_begin();
         field != struct_type->field_end(); field++) {
        nl::json field_metadata;
        std::string field_name = (*field)->getNameAsString();
        field_metadata["name"] = field_name;
        clang::QualType field_type = (*field)->getTypeSourceInfo()->getType();
        field_metadata["type"]["base_type"] = field_type.getAsString();
        llvm::errs() << "Base Type: " << field_type.getAsString() << '\n';
        field_metadata["type"]["type_kind"] = "Built-in";
        if (field_type.hasQualifiers())
            field_metadata["type"]["qualifiers"] =
                field_type.getQualifiers().getAsString();
        llvm::errs() << sourceDump(sm, lang_opt, *field) << '\n';
        metadata["fields"].push_back(field_metadata);
    }

    // ctors
    for (auto ctor = struct_type->ctor_begin(); ctor != struct_type->ctor_end();
         ctor++) {
        nl::json ctor_decl;
        for (auto param_idx = 0; param_idx < (*ctor)->getNumParams();
             param_idx++) {
            clang::ParmVarDecl *param = (*ctor)->getParamDecl(param_idx);
            ctor_decl["params"].push_back(sourceDump(sm, lang_opt, param));
        }
        auto body = (*ctor)->getBody();
        llvm::errs() << sourceDump(sm, lang_opt, body) << '\n';
        ctor_decl["body"] = sourceDump(sm, lang_opt, body);
        metadata["ctors"].push_back(ctor_decl);
    }
    m_function_metadata["records"].push_back(metadata);
    return true;
}

bool SpmdfyAction::enumType(const mat::MatchFinder::MatchResult &result) {
    llvm::StringRef ref("EnumType");
    auto *enum_type = result.Nodes.getNodeAs<clang::EnumDecl>(ref);
    if (!enum_type) {
        return false;
    }
    nl::json metadata;
    clang::SourceManager &sm = *result.SourceManager;
    clang::LangOptions lang_opt;
    llvm::errs() << "EnumType\n";
    llvm::errs() << enum_type->getNameAsString() << '\n';
    metadata["name"] = enum_type->getNameAsString();
    for (auto e = enum_type->enumerator_begin();
         e != enum_type->enumerator_end(); e++) {
        nl::json field;
        llvm::errs() << sourceDump(sm, lang_opt, e) << '\n';
        field["name"] = (*e)->getNameAsString();
        auto init_expr = (*e)->getInitExpr();
        if (init_expr) {
            field["init"] = sourceDump(sm, lang_opt, init_expr);
            llvm::errs() << "Init Expr: " << sourceDump(sm, lang_opt, init_expr)
                         << '\n';
        } else
            field["init"] =
                std::to_string((int)*(*e)->getInitVal().getRawData());
        metadata["fields"].push_back(field);
    }
    m_function_metadata["enum"].push_back(metadata);
    return true;
}

void SpmdfyAction::run(const mat::MatchFinder::MatchResult &result) {
    m_stmt_visitor = new SpmdfyStmtVisitor(*result.SourceManager);
    if (globalDeclarations(result))
        return;
    if (cudaKernelFunction(result))
        return;
    if (cudaDeviceFunction(result))
        return;
    if (structType(result))
        return;
    if (enumType(result))
        return;
}

} // namespace spmdfy