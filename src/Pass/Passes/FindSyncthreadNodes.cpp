#include <spmdfy/Pass/Passes/FindSyncthreadNodes.hpp>

namespace spmdfy {

namespace pass {

#define CFGNODE_DEF_VISITOR(NODE, NAME)                                        \
    auto FindSyncthreadNodes::Visit##NODE##Node(cfg::NODE##Node *NAME)->bool

#define CASTAS(TYPE, NODE) dynamic_cast<TYPE>(NODE)

bool findSyncthreadNodes(SpmdTUTy &spmd_tu, clang::ASTContext &ast_context,
                         Workspace &workspace) {
    FindSyncthreadNodes finder(spmd_tu, ast_context, workspace);
    for (auto node : spmd_tu) {
        SPMDFY_INFO("Visiting Node {}", node->getNodeTypeName());
        finder.Visit(node);
    }
    SPMDFY_INFO("Number of Synthread Nodes Detected: {}",
                workspace.syncthrds_queue.size());
    return false;
}

CFGNODE_DEF_VISITOR(KernelFunc, kernel) {
    cfg::CFGNode *curr_node = kernel->getNext();
    while (curr_node->getNodeType() != cfg::CFGNode::Exit) {
        Visit(curr_node);
        if (curr_node->getNodeType() == cfg::CFGNode::IfStmt) {
            if (CASTAS(cfg::IfStmtNode *, curr_node)) {
                curr_node = CASTAS(cfg::IfStmtNode *, curr_node)->getReconv();
            }
        }
        if (curr_node->getNodeType() == cfg::CFGNode::ForStmt) {
            if (CASTAS(cfg::ForStmtNode *, curr_node)) {
                curr_node = CASTAS(cfg::ForStmtNode *, curr_node)->getReconv();
            }
        }
        curr_node = curr_node->getNext();
    }
    return true;
}

CFGNODE_DEF_VISITOR(IfStmt, ifstmt) {
    auto if_stmt = ifstmt->getIfStmt();
    for (auto curr_node = ifstmt->getNext();
         curr_node->getNodeType() != cfg::CFGNode::Reconv;
         curr_node = curr_node->getNext()) {
        Visit(curr_node);
        if (curr_node->getNodeType() == cfg::CFGNode::IfStmt) {
            if (CASTAS(cfg::IfStmtNode *, curr_node)) {
                curr_node = CASTAS(cfg::IfStmtNode *, curr_node)->getReconv();
                continue;
            }
        }
    }
    for (auto curr_node = ifstmt->getFalseBlock();
         curr_node->getNodeType() != cfg::CFGNode::Reconv;
         curr_node = curr_node->getNext()) {
        Visit(curr_node);
    }
    return true;
}

CFGNODE_DEF_VISITOR(ForStmt, forstmt) {
    auto for_stmt = forstmt->getForStmt();
    for (auto curr_node = forstmt->getNext();
         curr_node->getNodeType() != cfg::CFGNode::Reconv;
         curr_node = curr_node->getNext()) {
        Visit(curr_node);
        if (curr_node->getNodeType() == cfg::CFGNode::IfStmt) {
            if (CASTAS(cfg::IfStmtNode *, curr_node)) {
                curr_node = CASTAS(cfg::IfStmtNode *, curr_node)->getReconv();
                continue;
            }
        }
    }
    return true;
}

CFGNODE_DEF_VISITOR(Internal, internal) {
    const std::string &node_name = internal->getInternalNodeName();
    SPMDFY_INFO("CodeGen InternalNode {} of type {}", internal->getName(), node_name);
    if (node_name == "CallExpr") {
        auto call_expr = internal->getInternalNodeAs<const clang::CallExpr>();
        if(call_expr->getDirectCallee()->getNameAsString() == "__syncthreads"){
            SPMDFY_INFO("Detected synthreads after {}", internal->getPrevious()->getName());
            m_workspace.syncthrds_queue.push(internal);
        }
    }
    return true;
}

} // namespace pass

} // namespace spmdfy