#include <spmdfy/SpmdfyAction.hpp>

namespace spmdfy {
std::string sourceDump(const clang::SourceManager &sm,
                       const clang::LangOptions &lang_opt,
                       const clang::SourceLocation &begin,
                       const clang::SourceLocation &end) {
    clang::SourceLocation e(
        clang::Lexer::getLocForEndOfToken(end, 0, sm, lang_opt));
    clang::SourceLocation b(
        clang::Lexer::GetBeginningOfToken(begin, sm, lang_opt));
    if ((sm.getCharacterData(e) - sm.getCharacterData(b)) < 1) {
        llvm::errs() << "Cannot dump source\n";
        return "";
    }
    return std::string(sm.getCharacterData(begin),
                       (sm.getCharacterData(e) - sm.getCharacterData(b)));
}

clang::Stmt *
SpmdfyStmtVisitor::VisitCompoundStmt(clang::CompoundStmt *cpmd_stmt) {
    llvm::errs() << "Visiting Compound Statment\n";
    for (auto i = cpmd_stmt->body_begin(); i != cpmd_stmt->body_end(); i++) {
        if (Visit(*i))
            continue;
    }
    return cpmd_stmt;
}
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
    for (auto param_idx = 0; param_idx < kernel_function->getNumParams();
         param_idx++) {
        metadata["params"].push_back(
            sourceDump(sm, lang_opt, kernel_function->getParamDecl(param_idx)));
    }
    
    clang::Stmt *body = kernel_function->getBody();
    if (body) {
        stmt_visitor->Visit(body);
    }
    m_function_metadata[name] = metadata;
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
    metadata["return_type"] = device_function->getReturnType().getAsString();

    // 4. Params
    metadata["params"] = {};
    for (auto param_idx = 0; param_idx < device_function->getNumParams();
         param_idx++) {
        metadata["params"].push_back(
            sourceDump(sm, lang_opt, device_function->getParamDecl(param_idx)));
    }
    clang::Stmt *body = device_function->getBody();
    if (body) {
        stmt_visitor->Visit(body);
    }
    m_function_metadata[name] = metadata;
    return true;
}

void SpmdfyAction::run(const mat::MatchFinder::MatchResult &result) {
    stmt_visitor = new SpmdfyStmtVisitor(*result.SourceManager);
    if (cudaKernelFunction(result))
        return;
    if (cudaDeviceFunction(result))
        return;
}

} // namespace spmdfy