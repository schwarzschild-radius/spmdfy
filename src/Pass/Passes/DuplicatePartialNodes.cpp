#include <spmdfy/Pass/Passes/DuplicatePartialNodes.hpp>

namespace spmdfy {

namespace pass {

#define CFGNODE_DEF_VISITOR(NODE, NAME)                                        \
    auto DuplicatePartialNodes::Visit##NODE##Node(cfg::NODE##Node *NAME)->bool

#define CASTAS(TYPE, NODE) dynamic_cast<TYPE>(NODE)

bool duplicatePartialNodes(SpmdTUTy &spmd_tu, clang::ASTContext &ast_context,
                           Workspace &workspace) {
    DuplicatePartialNodes finder(spmd_tu, ast_context, workspace);
    finder.HandleSpmdTU(spmd_tu);
    return false;
}

static std::string getKernelNodeName(cfg::CFGNode *node) {
    auto curr_node = node;
    while (curr_node->getNodeType() != cfg::CFGNode::KernelFunc) {
        curr_node = curr_node->getPrevious();
    }
    return curr_node->getName();
}

auto DuplicatePartialNodes::duplicateInternalNode(cfg::InternalNode * node) -> cfg::InternalNode* {
    auto duplicate = new cfg::InternalNode(m_ast_context, node->getInternalNode());
    return duplicate;
}

CFGNODE_DEF_VISITOR(ISPCBlock, block) {
    auto kernel = getKernelNodeName(block);
    cfg::CFGNode* hook = block;
    for(auto var : m_workspace.partial_nodes[kernel]){
        hook = hook->splitEdge(duplicateInternalNode(var));
    }
    return true;
}

} // namespace pass

} // namespace spmdfy