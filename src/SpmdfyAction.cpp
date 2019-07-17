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
        auto body = (*ctor)->getBody();
        llvm::errs() << sourceDump(sm, lang_opt, body) << '\n';
        for (auto param_idx = 0; param_idx < (*ctor)->getNumParams();
         param_idx++) {
            clang::ParmVarDecl* param = (*ctor)->getParamDecl(param_idx);
            ctor_decl["params"].push_back(sourceDump(sm, lang_opt, param));
        }
        metadata["ctors"].push_back(ctor_decl);
    }
    m_function_metadata["records"].push_back(metadata);
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
}

} // namespace spmdfy