#include <spmdfy/Pass/Passes/InsertISPCNodes.hpp>

namespace spmdfy {

namespace pass {

#define CASTAS(TYPE, NODE) dynamic_cast<TYPE>(NODE)

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

auto walkBackTill(cfg::CFGNode * node) -> cfg::CFGNode *{
    auto curr_node = node->getPrevious();
    while(true){
        SPMDFY_INFO("Walking back");
        switch(curr_node->getNodeType()){
            case cfg::CFGNode::Reconv:
                curr_node = CASTAS(cfg::ReconvNode*, curr_node)->getBack()->getPrevious();
            break;
            case cfg::CFGNode::IfStmt:
            case cfg::CFGNode::ForStmt:
            case cfg::CFGNode::KernelFunc:
                return curr_node;
            default:
                curr_node = curr_node->getPrevious();
        }
    }
    return nullptr;
}

auto InsertISPCNode::handleKernelFunc(cfg::KernelFuncNode * kernel) -> bool{
    kernel->splitEdge(new cfg::ISPCGridNode());
    auto curr_node = kernel->getNext();
    auto sync_node = m_workspace.syncthrds_queue.front();
    auto block_node = walkBackTill(sync_node);
    if(block_node->getNodeType() == cfg::CFGNode::KernelFunc){
        
    }
    SPMDFY_INFO("BlockNode : {}", block_node->getName());
    curr_node = curr_node->getPrevious();
    curr_node->splitEdge(new cfg::ISPCGridExitNode());
    return false;
}

} // namespace pass

} // namespace spmdfy