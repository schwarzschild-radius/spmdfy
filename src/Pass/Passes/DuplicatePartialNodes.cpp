#include <spmdfy/Pass/Passes/DuplicatePartialNodes.hpp>

namespace spmdfy {

namespace pass {

#define CASTAS(TYPE, NODE) dynamic_cast<TYPE>(NODE)

auto duplicateInternalNode(clang::ASTContext &ast_context,
                           cfg::InternalNode *node) -> cfg::InternalNode * {
    auto duplicate =
        new cfg::InternalNode(ast_context, node->getInternalNode());
    return duplicate;
}

bool duplicatePartialNodes(SpmdTUTy &spmd_tu, clang::ASTContext &ast_context,
                           Workspace &workspace) {
    for (auto decl : spmd_tu) {
        if (ISNODE(decl, cfg::CFGNode::KernelFunc)) {
            auto name = decl->getName();
            SPMDFY_INFO("[] Visting KernelFuncNode {}", name);
            auto &partial_nodes = workspace.partial_nodes[name];
            int block_count = 0;
            for (auto curr_node = decl->getNext();
                 !(ISNODE(curr_node, cfg::CFGNode::Exit));
                 curr_node = curr_node->getNext()) {
                if (ISNODE(curr_node, cfg::CFGNode::ISPCBlock)) {
                    block_count++;
                    for (auto i = 0; i < block_count; i++) {
                        for (auto var : partial_nodes[i]) {
                            SPMDFY_INFO("[DuplicatePartial Nodes] Inserting {}",
                                        var->getName());
                            curr_node = curr_node->splitEdge(
                                duplicateInternalNode(ast_context, var));
                        }
                    }
                }
            }
        }
    }
    return false;
}

} // namespace pass

} // namespace spmdfy