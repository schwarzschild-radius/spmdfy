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
                          mat::isExpansionInMainFile(), mat::isDefinition())
            .bind("cudaDeviceFunction"),
        this);
    // match global decls
    m_finder->addMatcher(
        mat::varDecl(mat::anyOf(mat::hasParent(mat::translationUnitDecl()),
                                mat::hasParent(mat::namespaceDecl())),
                     mat::isExpansionInMainFile())
            .bind("globalDeclarations"),
        this);
    // match struct types
    m_finder->addMatcher(
        mat::cxxRecordDecl(mat::isStruct(), mat::isExpansionInMainFile(),
                           mat::has(mat::cxxConstructorDecl(
                               mat::hasAttr(clang::attr::CUDADevice))))
            .bind("structType"),
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

    clang::Stmt *body = kernel_function->getBody();
    if (body) {
        stmt_visitor->Visit(body);
    }
    metadata["body"] = stmt_visitor->getFunctionBody();
    metadata["shmem"] = stmt_visitor->getSharedMem();
    m_function_metadata["function"][name] = metadata;
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
        stmt_visitor->Visit(body);
    }
    m_function_metadata["function"][name] = metadata;
    return true;
}

bool SpmdfyAction::globalDeclarations(
    const mat::MatchFinder::MatchResult &result) {
    llvm::StringRef ref("globalDeclarations");
    auto *global_variable = result.Nodes.getNodeAs<clang::VarDecl>(ref);
    if (!global_variable) {
        return false;
    }
    llvm::errs() << ref.str() << '\n';
    clang::SourceManager &sm = *result.SourceManager;
    clang::LangOptions lang_opt;
    if (global_variable->hasAttr<clang::CUDAConstantAttr>()) {
        llvm::errs() << sourceDump(sm, lang_opt,
                                   global_variable->getTypeSpecStartLoc(),
                                   global_variable->getSourceRange().getEnd())
                     << '\n';
        m_function_metadata["globals"].push_back(
            sourceDump(sm, lang_opt, global_variable->getTypeSpecStartLoc(),
                       global_variable->getSourceRange().getEnd()) +
            ";");
    } else {
        llvm::errs() << sourceDump(sm, lang_opt,
                                   global_variable->getSourceRange().getBegin(),
                                   global_variable->getSourceRange().getEnd())
                     << '\n';
        llvm::errs() << global_variable->getNameAsString() << '\n';
        m_function_metadata["globals"].push_back(
            sourceDump(sm, lang_opt,
                       global_variable->getSourceRange().getBegin(),
                       global_variable->getSourceRange().getEnd()) +
            ";");
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
    llvm::errs() << "Matcher: structType\n";
    llvm::errs() << struct_type->getNameAsString() << '\n';
    llvm::errs() << sourceDump(sm, lang_opt, struct_type) << '\n';
    for(auto i = struct_type->field_begin(); i != struct_type->field_end(); i++){
        llvm::errs() << sourceDump(sm, lang_opt, *i) << '\n';
    }
    std::string struct_name = struct_type->getNameAsString();
    auto& struct_handler = m_function_metadata["structs"][struct_name];
    struct_handler = {};
    for(auto i = struct_type->field_begin(); i != struct_type->field_end(); i++){
        struct_handler["fields"].push_back(sourceDump(sm, lang_opt, *i));
    }
    return true;
}

void SpmdfyAction::run(const mat::MatchFinder::MatchResult &result) {
    stmt_visitor = new SpmdfyStmtVisitor(*result.SourceManager);
    if (globalDeclarations(result))
        return;
    if (cudaKernelFunction(result))
        return;
    if (cudaDeviceFunction(result))
        return;
    if (structType(result))
            return;
}

} // namespace spmdfy