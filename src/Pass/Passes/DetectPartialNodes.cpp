#include <spmdfy/Pass/Passes/DetectPartialNodes.hpp>

namespace spmdfy {

namespace pass {

#define CASTAS(TYPE, NODE) dynamic_cast<TYPE>(NODE)

bool handleKernelFunc(cfg::KernelFuncNode *kernel, clang::ASTContext &context,
                      Workspace &workspace) {
    int curr_block = -1;
    auto& partial_node = workspace.partial_nodes[kernel->getName()];
    for (auto curr_node = kernel->getNext();
         !(ISNODE(curr_node, cfg::CFGNode::Exit));
         curr_node = curr_node->getNext()) {
        if (ISNODE(curr_node, cfg::CFGNode::ISPCBlock)) {
            curr_block++;
        } else if (ISNODE(curr_node, cfg::CFGNode::Internal)) {
            auto internal = CASTAS(cfg::InternalNode *, curr_node);
            if (internal->getName() == "Var") {
                auto var_decl =
                    internal->getInternalNodeAs<const clang::VarDecl>();
                if (!var_decl->hasAttr<clang::CUDASharedAttr>()) {
                    SPMDFY_INFO("[DetectPartialNodes] Detected Function Scope "
                                "variable {} of block scope {}",
                                internal->getName(), curr_block);
                    partial_node[curr_block].push_back(internal);
                    cfg::rmCFGNode(internal);
                }
            }
        } else if (auto cond_node = CASTAS(cfg::ConditionalNode *, curr_node);
                   cond_node) {
            curr_node = cond_node->getReconv();
        }
    }
    return false;
}

bool detectPartialNodes(SpmdTUTy &spmd_tu, clang::ASTContext &ast_context,
                        Workspace &workspace) {
    for (auto decl : spmd_tu) {
        SPMDFY_INFO("[DetectPartialNodes] Visiting Kernel Func");
        if (ISNODE(decl, cfg::CFGNode::KernelFunc)) {
            handleKernelFunc(CASTAS(cfg::KernelFuncNode *, decl), ast_context,
                             workspace);
        }
    }
    return false;
}

} // namespace pass

} // namespace spmdfy