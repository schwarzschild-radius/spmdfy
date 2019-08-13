#ifndef SPMDFY_CFGGENERATOR_HPP
#define SPMDFY_CFGGENERATOR_HPP

#include <spmdfy/Generator/Generator.hpp>
#include <spmdfy/Generator/CFGGenerator/ConstructCFG.hpp>
#include <spmdfy/Generator/CFGGenerator/CFG.hpp>
#include <spmdfy/Logger.hpp>
#include <spmdfy/utils.hpp>

#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

#include <clang/Analysis/CFG.h>

namespace spmdfy {

class CFGGenerator : public ISPCGenerator {
  public:
    enum class CUDAExprCategory { Partial, Grid, Block };
    using VariableMap =
        std::unordered_map<CUDAExprCategory, std::vector<std::string>>;

    CFGGenerator(clang::ASTContext &context, std::ostringstream &file_writer)
        : m_context(context), m_sm(context.getSourceManager()),
          m_lang_opts(context.getLangOpts()), m_file_writer(file_writer) {
        m_lang_opts.CPlusPlus = true;
        m_lang_opts.Bool = true;
    }
    auto handleTranslationUnit(clang::ASTContext &) -> bool override;
    auto handleFunctionDecl(clang::FunctionDecl *func_decl) -> std::string;

  private:
    clang::ASTContext &m_context;
    clang::SourceManager &m_sm;
    clang::LangOptions m_lang_opts;
    std::ostringstream &m_file_writer;
};

} // namespace spmdfy

#endif