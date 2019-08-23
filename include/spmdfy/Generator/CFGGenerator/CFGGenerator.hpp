#ifndef SPMDFY_CFGGENERATOR_HPP
#define SPMDFY_CFGGENERATOR_HPP

#include <spmdfy/CFG/CFG.hpp>
#include <spmdfy/Generator/CFGGenerator/CFGCodeGen.hpp>
#include <spmdfy/Generator/CFGGenerator/ConstructCFG.hpp>
#include <spmdfy/Generator/Generator.hpp>
#include <spmdfy/Logger.hpp>
#include <spmdfy/Pass/PassManager.hpp>
#include <spmdfy/utils.hpp>

#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

#include <clang/Analysis/CFG.h>

namespace spmdfy {
/**
 * \class CFGGenerator
 * \ingroup CodeGen
 *
 * \brief An ISPC generator that drives CFG Code generation. 
 * The following steps are executed in order
 * 1. CFG construction
 * 2. Running Pass sequence on CFG
 * 3. Code generation of CFG
 *
 * */
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
    // AST specific variables
    clang::ASTContext &m_context;
    clang::SourceManager &m_sm;
    clang::LangOptions m_lang_opts;

    std::ostringstream &m_file_writer;

    // CFG specific variables
    std::vector<cfg::CFGNode *> m_spmd_tutbl;
};

} // namespace spmdfy

#endif