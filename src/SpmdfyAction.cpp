#include <spmdfy/SpmdfyAction.hpp>

namespace spmdfy {

auto SpmdfyAction::CreateASTConsumer(clang::CompilerInstance &Compiler,
                                     llvm::StringRef InFile)
    -> std::unique_ptr<clang::ASTConsumer> {
    return std::unique_ptr<clang::ASTConsumer>(
        new SpmdfyConsumer(&Compiler.getASTContext(), m_file_writer));
}

auto SpmdfyConsumer::HandleTranslationUnit(clang::ASTContext &m_context)
    -> void {
    if (gen->handleTranslationUnit(m_context)) {
        SPMDFY_ERROR("Cannot Parse Translation Unit");
    }
}

auto SpmdfyFrontendActionFactory::create() -> clang::FrontendAction *{
    return action;
}

} // namespace spmdfy