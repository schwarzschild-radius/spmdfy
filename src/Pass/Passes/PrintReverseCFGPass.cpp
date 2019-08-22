#include <spmdfy/Pass/Passes/PrintReverseCFGPass.hpp>

namespace spmdfy {

namespace pass {

#define CASTAS(TYPE, NODE) dynamic_cast<TYPE>(NODE)

bool print_reverse_cfg_pass(SpmdTUTy &spmd_tu, clang::ASTContext &ast_context,
                            Workspace &workspace) {
    // 1. Getting the exit node
    for (auto node : spmd_tu) {
        SPMDFY_INFO("[PrintReverseCFGPass] Visiting Node {}",
                    node->getNodeTypeName());
        if (node->getNodeType() == cfg::CFGNode::KernelFunc) {
            cfg::CFGNode *curr_node = (CASTAS(cfg::KernelFuncNode*, node))->getExit();
            while (curr_node->getNodeType() != cfg::CFGNode::KernelFunc) {
                SPMDFY_INFO("[PrintReverseCFGPass] Visiting Node {}, {}",
                            curr_node->getSource(), curr_node->getNodeTypeName());
                curr_node = curr_node->getPrevious();
            }
        }
    }
    return false;
}

} // namespace pass

} // namespace spmdfy