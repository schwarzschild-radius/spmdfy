#include <spmdfy/Pass/Passes/FindSyncthreadNodes.hpp>

namespace spmdfy {

namespace pass {

#define CFGNODE_DEF_VISITOR(NODE, NAME)                                        \
    auto FindSyncthreadNodes::Visit##NODE##Node(cfg::NODE##Node *NAME)->bool

#define CASTAS(TYPE, NODE) dynamic_cast<TYPE>(NODE)

bool findSyncthreadNodes(SpmdTUTy &spmd_tu, clang::ASTContext &ast_context,
                         Workspace &workspace) {
    FindSyncthreadNodes finder(spmd_tu, ast_context, workspace);
    finder.HandleSpmdTU(spmd_tu);
    SPMDFY_INFO("Number of Synthread Nodes Detected: {}",
                workspace.syncthrds_queue.size());
    return false;
}

CFGNODE_DEF_VISITOR(Internal, internal) {
    const std::string &node_name = internal->getInternalNodeName();
    SPMDFY_INFO("Visiting InternalNode {} of type {}", internal->getName(), node_name);
    if (node_name == "CallExpr") {
        auto call_expr = internal->getInternalNodeAs<const clang::CallExpr>();
        if(call_expr->getDirectCallee()->getNameAsString() == "__syncthreads"){
            SPMDFY_INFO("Detected synthreads after {}", internal->getPrevious()->getName());
            m_workspace.syncthrds_queue.push(internal);
        }
    }
    return true;
}

} // namespace pass

} // namespace spmdfy