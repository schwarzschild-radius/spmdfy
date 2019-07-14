#ifndef SPMDFY_STMT_VISITOR_HPP
#define SPMDFY_STMT_VISITOR_HPP

// clang headers
#include <clang/AST/StmtVisitor.h>

// llvm headers
#include <llvm/Support/Debug.h>

// standard headers
#include <string>
#include <vector>
#include <unordered_map>
#include <sstream>

// spmdfy headers
#include <spmdfy/utils.hpp>

// third_party headers
#include <nlohmann/json.hpp>

namespace nl = nlohmann;

namespace spmdfy {
class SpmdfyStmtVisitor
    : public clang::StmtVisitor<SpmdfyStmtVisitor, clang::Stmt *> {
  public:
    SpmdfyStmtVisitor(clang::SourceManager &sm) : m_sm(sm) {
        m_function_body[m_block] = {};
    }

    // getters
    nl::json getFunctionBody() { return m_function_body; }
    nl::json getSharedMem() { return m_shmem; }
    nl::json getExternSharedMem() { return m_extern_shmem; }
    nl::json getContext() { return m_context; }
    // visitors
    clang::Stmt *VisitCompoundStmt(clang::CompoundStmt *C);
    clang::Stmt *VisitCallExpr(clang::CallExpr *call_expr);
    clang::Stmt *VisitDeclStmt(clang::DeclStmt *decl_stmt);
    clang::Stmt *VisitForStmt(clang::ForStmt *for_stmt);
    clang::Stmt *VisitIfStmt(clang::IfStmt *if_stmt);

  private:
    clang::SourceManager &m_sm;
    clang::LangOptions m_lang_opt;
    nl::json m_function_body;
    nl::json m_shmem;
    nl::json m_extern_shmem;
    nl::json m_atomics;
    nl::json m_context = {};
    int m_scope = -1;
    int m_block = 0;
};

} // namespace spmdfy

#endif