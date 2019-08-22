#ifndef HOIST_SHMEM_NODES_HPP
#define HOIST_SHMEM_NODES_HPP

#include <spmdfy/CFG/RecursiveCFGVisitor.hpp>
#include <clang/AST/Expr.h>
#include <spmdfy/Pass/PassHandler.hpp>

namespace spmdfy {

namespace pass {

bool hoistShmemNodes(SpmdTUTy &, clang::ASTContext &, Workspace &);

PASS(hoistShmemNodes, hoist_shmem_nodes_pass_t);

#undef CFGNODE_VISITOR

} // namespace pass
} // namespace spmdfy

#endif