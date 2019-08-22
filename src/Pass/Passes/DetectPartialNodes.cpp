#include <spmdfy/Pass/Passes/DetectPartialNodes.hpp>

namespace spmdfy {

namespace pass {

#define CFGNODE_DEF_VISITOR(NODE, NAME)                                        \
    auto DetectPartialNodes::Visit##NODE##Node(cfg::NODE##Node *NAME)->bool

#define CASTAS(TYPE, NODE) dynamic_cast<TYPE>(NODE)

bool detectPartialNodes(SpmdTUTy &spmd_tu, clang::ASTContext &ast_context,
                        Workspace &workspace) {
    DetectPartialNodes finder(spmd_tu, ast_context, workspace);
    finder.HandleSpmdTU(spmd_tu);
    return false;
}

CFGNODE_DEF_VISITOR(KernelFunc, kernel) {
    auto kernel_name = kernel->getName();
    SPMDFY_INFO("Detecting Partial Node of {} : ", kernel_name);
    auto &v = m_workspace.partial_nodes[kernel_name];
    for (auto curr_node = kernel->getNext();
         !ISNODE(curr_node, cfg::CFGNode::Exit);
         curr_node = curr_node->getNext()) {
        if (ISNODE(curr_node, cfg::CFGNode::Internal)) {
            auto internal = CASTAS(cfg::InternalNode *, curr_node);
            if (internal->getInternalNodeName() == "Var") {
                auto var_decl = internal->getInternalNodeAs<const clang::VarDecl>();
                if(!var_decl->hasAttr<clang::CUDASharedAttr>()){
                    v.push_back(internal);
                }
            }
        }
    }

    for (auto var : v) {
        SPMDFY_INFO("PartialNode: {}", var->getName());
    }
    return false;
}

} // namespace pass

} // namespace spmdfy