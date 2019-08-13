#include <spmdfy/Generator/CFGGenerator/CFGGenerator.hpp>

namespace spmdfy {

auto CFGGenerator::handleTranslationUnit(clang::ASTContext &context) -> bool {
    clang::DeclContext *traverse_decl =
        llvm::dyn_cast<clang::DeclContext>(context.getTranslationUnitDecl());
    if (!traverse_decl) {
        SPMDFY_ERROR("Cannot Traverse Decl");
        return true;
    }
    for (auto D : traverse_decl->decls()) {
        if (!isExpansionInMainFile(m_sm, D)) {
            continue;
        }
        switch (D->getKind()) {
        case clang::Decl::Function:
            handleFunctionDecl(llvm::cast<clang::FunctionDecl>(D));
            break;
        default:
            SPMDFY_ERROR("Declaration not supported yet!");
            break;
        }
    }
    SPMDFY_INFO("Translation Unit:\n {}", m_file_writer.str());
    return false;
}

auto CFGGenerator::handleFunctionDecl(clang::FunctionDecl *func_decl)
    -> std::string {
    SPMDFY_INFO("Visiting FunctionDecl {}", func_decl->getNameAsString());
    VariableMap var_map;
    std::ostringstream func_gen;
    auto cfg = clang::CFG::buildCFG(
        func_decl, llvm::cast<clang::CompoundStmt>(func_decl->getBody()),
        &m_context, clang::CFG::BuildOptions());
    if (cfg->isLinear()) {
        SPMDFY_INFO("No Control flow statements");
        SPMDFY_INFO("We can you the simple generator");
    }

    clang::CompoundStmt *cpmd_stmt =
        llvm::cast<clang::CompoundStmt>(func_decl->getBody());

    ConstructSpmdCFG test(m_context, cpmd_stmt);
    test.get();

    return func_gen.str();
}

} // namespace spmdfy