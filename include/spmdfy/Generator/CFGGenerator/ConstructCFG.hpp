#ifndef SPMDFY_CONSTRUCTCFG_HPP
#define SPMDFY_CONSTRUCTCFG_HPP

#include <clang/AST/RecursiveASTVisitor.h>
#include <spmdfy/Generator/CFGGenerator/CFG.hpp>
#include <spmdfy/Logger.hpp>
#include <spmdfy/utils.hpp>

#include <vector>

namespace spmdfy {
class ConstructSpmdCFG : public clang::RecursiveASTVisitor<ConstructSpmdCFG> {
  public:
    ConstructSpmdCFG(clang::ASTContext &context)
        : m_context(context),
          m_sm(context.getSourceManager()), m_lang_opts(context.getLangOpts()) {
    }

    // Added nodes
    auto add(const clang::VarDecl*) -> bool;
    auto add(const clang::FunctionDecl*) -> bool;
    auto add(const clang::CXXRecordDecl*) -> bool;

    auto splitEdge(CFG::CFGNode * node) -> bool;

    auto get() -> std::vector<CFG::CFGNode*>;
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
    CFG::CFGNode *m_curr_node;
    std::vector<const clang::Decl*> m_cpp_tutbl;
    std::vector<CFG::CFGNode*> m_spmdfy_tutbl;
};
} // namespace spmdfy

#endif