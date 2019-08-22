#include <spmdfy/Pass/Passes/HoistShmemNodes.hpp>

namespace spmdfy {

namespace pass {

#define CASTAS(TYPE, NODE) dynamic_cast<TYPE>(NODE)

auto getISPCGridNode(cfg::KernelFuncNode *kernel){
    return kernel->getNext();
}

bool hoistShmemNodes(SpmdTUTy &spmd_tu, clang::ASTContext &ast_context,
                     Workspace &workspace) {
    for (auto node : spmd_tu) {
        SPMDFY_INFO("[HoistShmemNodes] Visiting Node {}", node->getNodeTypeName());
        if (node->getNodeType() == cfg::CFGNode::KernelFunc) {
            auto grid_node = node; //getISPCGridNode(CASTAS(cfg::KernelFuncNode*, node));
            auto& queue = workspace.shmem_queue;
            while(queue.size()){
                auto shmem_node = queue.front();
                cfg::rmCFGNode(shmem_node);
                grid_node->splitEdge(shmem_node);
                queue.pop();
            }
        }
    }
    return false;
}

} // namespace pass

} // namespace spmdfy