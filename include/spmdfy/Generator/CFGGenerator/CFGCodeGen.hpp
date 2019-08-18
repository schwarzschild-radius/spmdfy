#ifndef CFGCODEGEN_HPP
#define CFGCODEGEN_HPP

#include <spmdfy/CUDA2ISPC.hpp>
#include <spmdfy/CFG/CFG.hpp>
#include <spmdfy/Logger.hpp>
#include <spmdfy/utils.hpp>

#include <clang/AST/DeclVisitor.h>
#include <clang/AST/RecursiveASTVisitor.h>
#include <clang/AST/StmtVisitor.h>
#include <clang/AST/TypeVisitor.h>

#include <sstream>
#include <string>
#include <vector>

namespace spmdfy {
namespace codegen {

class CFGCodeGen : public clang::ConstDeclVisitor<CFGCodeGen, std::string>,
                   public clang::ConstStmtVisitor<CFGCodeGen, std::string>,
                   public clang::TypeVisitor<CFGCodeGen, std::string> {
    using clang::ConstDeclVisitor<CFGCodeGen, std::string>::Visit;
    using clang::ConstStmtVisitor<CFGCodeGen, std::string>::Visit;
    using clang::TypeVisitor<CFGCodeGen, std::string>::Visit;

  public:
    using OStreamTy = std::ostringstream;
    CFGCodeGen(clang::ASTContext &ast_context,
               const std::vector<cfg::CFGNode *> &node)
        : m_ast_context(ast_context), m_sm(ast_context.getSourceManager()),
          m_lang_opts(ast_context.getLangOpts()), m_node(node) {
        m_lang_opts.CPlusPlus = true;
        m_lang_opts.Bool = true;
    }
    auto get() -> std::string const;
    auto getFrom(cfg::CFGNode *) -> std::string const;
    auto traverseCFG() -> std::string const;

    // ispc code generators
    auto ispcCodeGen(cfg::KernelFuncNode *kernel) -> std::string;
    auto ispcCodeGen(cfg::InternalNode *internal) -> std::string;
    auto getISPCBaseType(std::string type) -> std::string;

    // ispc code gen vistiors
#define DECL_VISITOR(NODE)                                                     \
    auto Visit##NODE##Decl(const clang::NODE##Decl *)->std::string
#define STMT_VISITOR(NODE)                                                     \
    auto Visit##NODE##Stmt(const clang::NODE##Stmt *)->std::string
#define EXPR_VISITOR(NODE)                                                     \
    auto Visit##NODE##Expr(const clang::NODE##Expr *)->std::string
#define TYPE_VISITOR(NODE)                                                     \
    auto Visit##NODE##Type(const clang::NODE##Type *)->std::string

    DECL_VISITOR(Var);
    DECL_VISITOR(ParmVar);
    DECL_VISITOR(Function);

    auto VisitQualType(clang::QualType qual) -> std::string;
    TYPE_VISITOR(Builtin);
    TYPE_VISITOR(Pointer);
    TYPE_VISITOR(Record);
    TYPE_VISITOR(IncompleteArray);

  private:
    // AST specific variables
    clang::ASTContext &m_ast_context;
    clang::SourceManager &m_sm;
    clang::LangOptions m_lang_opts;

    cfg::CFGNode::Context m_tu_context;

    const std::vector<cfg::CFGNode *> m_node;
};

} // namespace codegen
} // namespace spmdfy

#endif