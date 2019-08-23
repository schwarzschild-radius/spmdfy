#ifndef SPMDFY_CONSTRUCTCFG_HPP
#define SPMDFY_CONSTRUCTCFG_HPP

#include <clang/AST/RecursiveASTVisitor.h>
#include <spmdfy/CFG/CFG.hpp>
#include <spmdfy/Logger.hpp>
#include <spmdfy/utils.hpp>

#include <vector>

namespace spmdfy {

/**
 * \class ConstructCFG
 * \ingroup CodeGen
 *
 * \brief Uses RecurisveASTVisitor to Construct a CFG of a Kernel Function(can
 * be extended for device functions)
 *
 * */
class ConstructSpmdCFG : public clang::RecursiveASTVisitor<ConstructSpmdCFG> {
  public:
    ConstructSpmdCFG(clang::ASTContext &context)
        : m_context(context), m_sm(context.getSourceManager()),
          m_lang_opts(context.getLangOpts()) {}

    // Added nodes
    auto add(const clang::VarDecl *) -> bool;
    auto add(const clang::FunctionDecl *) -> bool;
    auto add(const clang::CXXRecordDecl *) -> bool;

    auto splitEdge(cfg::CFGNode *node) -> bool;

    auto get() -> std::vector<cfg::CFGNode *>;
// visitors
#define DEF_VISITOR(NODE, BASE)                                                \
    auto Visit##NODE##BASE(clang::NODE##BASE *)->bool;
    DEF_VISITOR(Compound, Stmt);
    DEF_VISITOR(Decl, Stmt);
    DEF_VISITOR(For, Stmt);
    DEF_VISITOR(If, Stmt);

    DEF_VISITOR(Call, Expr);
    DEF_VISITOR(PseudoObject, Expr);

    DEF_VISITOR(CompoundAssign, Operator);
    DEF_VISITOR(Binary, Operator);

  private:
    // AST variables
    clang::ASTContext &m_context;
    clang::SourceManager &m_sm;
    clang::LangOptions m_lang_opts;

    // CFG variables
    size_t m_stmt_count = 0;
    cfg::CFGNode *m_curr_node;
    std::vector<const clang::Decl *> m_cpp_tutbl;
    std::vector<cfg::CFGNode *> m_spmdfy_tutbl;
};
} // namespace spmdfy

#endif