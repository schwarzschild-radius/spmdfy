#include <spmdfy/Pass/Passes/InsertISPCNodes.hpp>

namespace spmdfy {

namespace pass {

bool insertISPCNodes(SpmdTUTy& spmd_tu, clang::ASTContext& ast_context, Workspace& workspace) {
    InsertISPCNode inserter(spmd_tu, ast_context, workspace);
    for (auto node : spmd_tu) {
        SPMDFY_INFO("Visiting Node {}", node->getNodeTypeName());
        if(node->getNodeType() == cfg::CFGNode::KernelFunc){
            if(inserter.handleKernelFunc(dynamic_cast<cfg::KernelFuncNode*>(node))){
                SPMDFY_ERROR("Something is wrong");
                return true;
            }
        }
    }
    return false;
}

auto InsertISPCNode::handleKernelFunc(cfg::KernelFuncNode * kernel) -> bool{
    auto curr_node = kernel->getNext();
    kernel->splitEdge(new cfg::ISPCBlockNode());
    kernel->splitEdge(new cfg::ISPCGridNode());
    while(curr_node->getNodeType() != cfg::CFGNode::Exit){
        curr_node = curr_node->getNext();
    }
    curr_node = curr_node->getPrevious();
    curr_node->splitEdge(new cfg::ISPCGridExitNode());
    curr_node->splitEdge(new cfg::ISPCBlockExitNode());
    return false;
}

} // namespace pass

} // namespace spmdfy