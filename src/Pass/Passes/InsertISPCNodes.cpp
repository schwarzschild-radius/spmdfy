#include <spmdfy/Pass/Passes/InsertISPCNodes.hpp>
#include <tuple>

namespace spmdfy {

namespace pass {

#define CASTAS(TYPE, NODE) dynamic_cast<TYPE>(NODE)

bool insertISPCNodes(SpmdTUTy &spmd_tu, clang::ASTContext &ast_context,
                     Workspace &workspace) {
    InsertISPCNodes inserter(spmd_tu, ast_context, workspace);
    return !inserter.HandleSpmdTU(spmd_tu);
}

auto walkBackTill(cfg::CFGNode *node,
                  cfg::CFGNode::Node node_type = cfg::CFGNode::KernelFunc)
    -> std::tuple<cfg::CFGNode *, cfg::CFGNode::Node> {
    auto curr_node = node->getPrevious();
    while (true) {
        SPMDFY_INFO("Walking back");
        switch (curr_node->getNodeType()) {
        case cfg::CFGNode::Reconv:
            curr_node =
                CASTAS(cfg::ReconvNode *, curr_node)->getBack()->getPrevious();
            break;
        case cfg::CFGNode::IfStmt:
        case cfg::CFGNode::ForStmt:
        case cfg::CFGNode::ISPCBlock:
            return {curr_node, curr_node->getNodeType()};
        default:
            curr_node = curr_node->getPrevious();
        }
    }
    return {nullptr, cfg::CFGNode::Exit};
}

auto InsertISPCNodes::VisitKernelFuncNode(cfg::KernelFuncNode *kernel) -> bool {
    auto &syncthreads_queue = m_workspace.syncthreads_queue[kernel->getName()];
    // 1. Inserting GridNode
    auto grid_start = new cfg::ISPCGridNode();
    kernel->splitEdge(grid_start);

    // 2. Inserting BlockNode
    auto block_start = new cfg::ISPCBlockNode();
    grid_start->splitEdge(block_start);

    // 3. Getting sync_node
    if (syncthreads_queue.empty()) {
        auto last_node = kernel->getExit()->getPrevious();
        last_node->splitEdge(new cfg::ISPCBlockExitNode())
            ->splitEdge(new cfg::ISPCGridExitNode());
        return false;
    }

    while (syncthreads_queue.size()) {
        auto sync_node = syncthreads_queue.front();
        syncthreads_queue.pop();

        // 4. Getting back node
        auto [back_node, back_node_type] = walkBackTill(sync_node);
        SPMDFY_INFO("BlockNode : {}", back_node->getName());

        auto sync_block_end = new cfg::ISPCBlockExitNode();
        auto sync_block_start = new cfg::ISPCBlockNode();
        auto sync_replace = new cfg::ISPCBlockExitNode();
        auto prev_for = back_node->getPrevious();
        auto prev_if = back_node->getPrevious();
        switch (back_node_type) {
        case cfg::CFGNode::ISPCBlock:
            sync_node->splitEdge(sync_block_end);
            sync_block_end->splitEdge(sync_block_start);
            break;
        case cfg::CFGNode::ForStmt:
            prev_for->splitEdge(sync_block_end);
            back_node->splitEdge(sync_block_start);
            sync_node->splitEdge(sync_replace);
            break;
        case cfg::CFGNode::IfStmt:
            prev_if->splitEdge(sync_block_end);
            back_node->splitEdge(sync_block_start);
            sync_node->splitEdge(sync_replace);
            break;
        default:
            SPMDFY_ERROR("Wrong Back Node");
        }
        cfg::rmCFGNode(sync_node);
    }

    // A walk from top
    cfg::CFGNode *curr_node = block_start;
    for (; !ISNODE(curr_node, cfg::CFGNode::Exit);
         curr_node = curr_node->getNext()) {
        if (ISNODE(curr_node, cfg::CFGNode::ISPCBlockExit)) {
            if (CASTAS(cfg::ConditionalNode *, curr_node->getNext())) {
                curr_node = CASTAS(cfg::ConditionalNode *, curr_node->getNext())
                                ->getReconv();
            } else if (ISNODE(curr_node->getNext(), cfg::CFGNode::ISPCBlock)) {
                continue;
            }
            curr_node = curr_node->splitEdge(new cfg::ISPCBlockNode());
        }
    }
    curr_node = curr_node->getPrevious();
    curr_node = curr_node->splitEdge(new cfg::ISPCBlockExitNode());
    curr_node->splitEdge(new cfg::ISPCGridExitNode());
    return false;
}

} // namespace pass

} // namespace spmdfy