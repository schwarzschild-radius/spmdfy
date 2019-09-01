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

auto SpmdfyFrontendActionFactory::create() -> clang::FrontendAction * {
    return action;
}

auto SpmdfyAction::InclusionDirective(
    clang::SourceLocation hash_loc, const clang::Token &include_token,
    llvm::StringRef filename, bool is_angled,
    clang::CharSourceRange filename_range, const clang::FileEntry *file,
    llvm::StringRef search_path, llvm::StringRef relative_path,
    const clang::Module *imported, clang::SrcMgr::CharacteristicKind FileType)
    -> void {
    clang::SourceManager &sm = getCompilerInstance().getSourceManager();
    if (!sm.isWrittenInMainFile(hash_loc)) {
        return;
    }
    const std::string cuda_header_ext = "cuh";
    if (std::equal(filename.end() - 3, filename.end(),
                   cuda_header_ext.begin())) {
        SPMDFY_INFO("Detected header {}", filename.str());
        m_file_writer << "#include \""
                      << std::string(filename.begin(), filename.end() - 4)
                      << ".ispc.h\""
                      << "\n";
    }
}

/// Fixing includes

class PPCallbackProxy : public clang::PPCallbacks {
  public:
    explicit PPCallbackProxy(SpmdfyAction &action) : m_action(action) {}
    void InclusionDirective(
        clang::SourceLocation hash_loc, const clang::Token &include_token,
        llvm::StringRef file_name, bool is_angled,
        clang::CharSourceRange filename_range, const clang::FileEntry *file,
        llvm::StringRef search_path, llvm::StringRef relative_path,
        const clang::Module *imported,
        clang::SrcMgr::CharacteristicKind FileType) override {
        m_action.InclusionDirective(
            hash_loc, include_token, file_name, is_angled, filename_range, file,
            search_path, relative_path, imported, FileType);
    }

  private:
    SpmdfyAction &m_action;
};

auto SpmdfyAction::ExecuteAction() -> void {
    SPMDFY_INFO("Executing Action");
    clang::Preprocessor &pp = getCompilerInstance().getPreprocessor();
    pp.addPPCallbacks(std::unique_ptr<clang::PPCallbacks>(
        new spmdfy::PPCallbackProxy(*this)));
    clang::ASTFrontendAction::ExecuteAction();
};

} // namespace spmdfy