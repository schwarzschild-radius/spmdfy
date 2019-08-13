#ifndef SPMDFY_CONSTRUCTCFG_HPP
#define SPMDFY_CONSTRUCTCFG_HPP

#include <clang/AST/RecursiveASTVisitor.h>
#include <spmdfy/Generator/CFGGenerator/CFG.hpp>
#include <spmdfy/Logger.hpp>
#include <spmdfy/utils.hpp>

namespace spmdfy {
class ConstructSpmdCFG : public clang::RecursiveASTVisitor<ConstructSpmdCFG> {
  public:
    ConstructSpmdCFG(clang::ASTContext &context, clang::CompoundStmt *cpmd_stmt)
        : m_cpmd_stmt(cpmd_stmt), m_context(context),
          m_sm(context.getSourceManager()), m_lang_opts(context.getLangOpts()) {
    }
    auto get() -> std::shared_ptr<CFGNode>;
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
    clang::CompoundStmt *m_cpmd_stmt;
    std::shared_ptr<CFGNode> m_entry;
    clang::ASTContext &m_context;
    clang::SourceManager &m_sm;
    clang::LangOptions m_lang_opts;
    size_t m_stmt_count = 0;
};
} // namespace spmdfy

#endif