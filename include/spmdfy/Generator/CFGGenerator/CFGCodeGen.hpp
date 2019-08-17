#ifndef CFGCODEGEN_HPP
#define CFGCODEGEN_HPP

#include <spmdfy/Generator/CFGGenerator/CFG.hpp>
#include <spmdfy/Logger.hpp>

#include <clang/AST/DeclVisitor.h>
#include <clang/AST/RecursiveASTVisitor.h>
#include <clang/AST/StmtVisitor.h>

#include <sstream>
#include <vector>
#include <string>

namespace spmdfy {
namespace codegen {

class CFGCodeGen : public clang::ConstDeclVisitor<CFGCodeGen, std::string>,
      public clang::ConstStmtVisitor<CFGCodeGen, std::string> {
    using clang::ConstDeclVisitor<CFGCodeGen, std::string>::Visit;
    using clang::ConstStmtVisitor<CFGCodeGen, std::string>::Visit;
  public:
    using OStreamTy = std::ostringstream;
    CFGCodeGen(clang::ASTContext &ast_context,
               const std::vector<CFG::CFGNode *> &node)
        : m_ast_context(ast_context), m_sm(ast_context.getSourceManager()),
          m_lang_opts(ast_context.getLangOpts()), m_node(node) {
        m_lang_opts.CPlusPlus = true;
        m_lang_opts.Bool = true;
    }
    auto get() -> std::string const;
    auto getFrom(CFG::CFGNode *) -> std::string const;
    auto traverseCFG() -> std::string const;

    // ispc code generators
    auto ispcCodeGen(CFG::KernelFuncNode *kernel) -> std::string;
    auto ispcCodeGen(CFG::InternalNode *internal) -> std::string;

    // ispc code gen vistiors
#define DECL_VISITOR(NODE)                                                     \
    auto Visit##NODE##Decl(const clang::NODE##Decl *)->std::string
#define STMT_VISITOR(NODE)                                                     \
    auto Visit##NODE##Stmt(const clang::NODE##Stmt *)->std::string

    DECL_VISITOR(Var);

  private:
    // AST specific variables
    clang::ASTContext &m_ast_context;
    clang::SourceManager &m_sm;
    clang::LangOptions m_lang_opts;

    const std::vector<CFG::CFGNode *> m_node;
};

} // namespace codegen
} // namespace spmdfy

#endif