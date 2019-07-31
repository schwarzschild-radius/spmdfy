#include <spmdfy/SpmdfyAction.hpp>

namespace spmdfy {

std::unique_ptr<clang::ASTConsumer> SpmdfyAction::newASTConsumer() {
    m_finder.reset(new clang::ast_matchers::MatchFinder);
    m_finder->addMatcher(
        mat::functionDecl(mat::hasAttr(clang::attr::CUDAGlobal),
                          mat::isDefinition())
            .bind("cudaKernelFunction"),
        this);
    m_finder->addMatcher(
        mat::functionDecl(mat::hasAttr(clang::attr::CUDADevice),
                          mat::isExpansionInMainFile(), mat::isDefinition())
            .bind("cudaDeviceFunction"),
        this);
    m_finder->addMatcher(
        mat::varDecl(mat::isExpansionInMainFile(), mat::hasGlobalStorage(),
                     mat::hasParent(mat::translationUnitDecl()))
            .bind("globalDeclarations"),
        this);
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
    clang::SourceManager &sm = *result.SourceManager;
    clang::LangOptions lang_opt = result.Context->getLangOpts();
    lang_opt.CPlusPlus = true;
    clang::PrintingPolicy pm(lang_opt);
    pm.Bool = true;
    nl::json metadata;
    std::string var_name = var_decl->getNameAsString();
    metadata["name"] = var_name;
    clang::QualType var_type = var_decl->getType();
    std::string var_type_str = var_type.getAsString(pm);
    if (var_type.hasQualifiers()) {
        metadata["qualifiers"] = var_type.getQualifiers().getAsString();
        var_type_str = var_type.getUnqualifiedType().getAsString();
    }
    if (g_SpmdfyTypeMap.find(var_type_str) != g_SpmdfyTypeMap.end()) {
        var_type_str = g_SpmdfyTypeMap.at(var_type_str);
    }
    metadata["type"]["base_type"] = var_type_str;
    metadata["type"]["is_built_in"] = true;
    if (const clang::Expr *var_init = var_decl->getInit(); var_init) {
        if (!var_type->isBuiltinType()) {
            metadata["type"]["is_built_in"] = false;
            var_init->dump();
            if (llvm::isa<clang::CXXConstructExpr>(var_init)) {
                    const clang::CXXConstructExpr * ctor_expr = llvm::cast<clang::CXXConstructExpr>(var_init);
                    std::string ctor_type;
                    std::vector<std::string> ctor_args;
                    for(int i = 0; i < ctor_expr->getNumArgs(); i++){
                        ctor_type += "_" + ctor_expr->getArg(i)->getType().getAsString();
                        ctor_args.push_back(sourceDump(sm, lang_opt, ctor_expr->getArg(i)));
                    }
                    std::string var_init_str = var_type_str + "_ctor" + ctor_type + "(";
                    var_init_str += strJoin(ctor_args.begin(), ctor_args.end());
                    var_init_str += ")";
                    var_type_str = var_type_str + "&";
                    metadata["init"] = var_init_str;
                    metadata["type"]["base_type"] = var_type_str;
            }
        } else {
            metadata["init"] = sourceDump(sm, lang_opt, var_init);
        }
    }
    m_function_metadata["globals"].push_back(metadata);
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
}

} // namespace spmdfy