#ifndef DUPLICATE_PARTIAL_NODES_HPP
#define DUPLICATE_PARTIAL_NODES_HPP

#include <clang/AST/Expr.h>
#include <spmdfy/CFG/RecursiveCFGVisitor.hpp>
#include <spmdfy/Pass/PassHandler.hpp>

namespace spmdfy {

namespace pass {

bool duplicatePartialNodes(SpmdTUTy &, clang::ASTContext &, Workspace &);

PASS(duplicatePartialNodes, duplicate_partial_nodes_pass_t);

#undef CFGNODE_VISITOR

} // namespace pass
} // namespace spmdfy

#endif