#include <spmdfy/Pass/Passes/LocateASTNodes.hpp>

namespace spmdfy {

namespace pass {

#define CFGNODE_DEF_VISITOR(NODE, NAME)                                        \
    auto LocateASTNodes::Visit##NODE##Node(cfg::NODE##Node *NAME)->bool

#define CASTAS(TYPE, NODE) dynamic_cast<TYPE>(NODE)

bool locateASTNodes(SpmdTUTy &spmd_tu, clang::ASTContext &ast_context,
                         Workspace &workspace) {
    LocateASTNodes finder(spmd_tu, ast_context, workspace);
    finder.HandleSpmdTU(spmd_tu);
    SPMDFY_INFO("Number of Synthread Nodes Detected: {}",
                workspace.syncthrds_queue.size());
    SPMDFY_INFO("Number of Shmem Nodes Detected: {}", workspace.shmem_queue.size());
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
    if (node_name == "Var"){
        auto var_decl = internal->getInternalNodeAs<const clang::VarDecl>();
        if(var_decl->hasAttr<clang::CUDASharedAttr>()){
            SPMDFY_INFO("Detected SharedMemory Nodes {}", internal->getPrevious()->getName());
            m_workspace.shmem_queue.push(internal);
        }
    }
    return true;
}

} // namespace pass

} // namespace spmdfy