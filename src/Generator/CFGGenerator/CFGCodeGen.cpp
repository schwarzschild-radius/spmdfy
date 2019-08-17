#include <spmdfy/Generator/CFGGenerator/CFGCodeGen.hpp>

namespace spmdfy {
namespace codegen {

auto CFGCodeGen::get() -> std::string const {
    SPMDFY_INFO("Generating Code\n");
    return traverseCFG();
}

auto CFGCodeGen::getFrom(CFG::CFGNode *) -> std::string const { return ""; }

auto CFGCodeGen::traverseCFG() -> std::string const {
    OStreamTy tu_gen;
    for(auto node : m_node){
        if(node->getNodeType() == CFG::CFGNode::KernelFunc){
            tu_gen << ispcCodeGen(dynamic_cast<CFG::KernelFuncNode*>(node));
        }
    }
    return tu_gen.str();
}

auto CFGCodeGen::ispcCodeGen(CFG::InternalNode* internal) -> std::string{
    SPMDFY_INFO("CodeGen InternalNode {}", internal->getName());
    OStreamTy internal_gen;
    const std::string& node_name = internal->getInternalNodeName();
    if(node_name == "Var"){
        Visit(internal->getInternalNode<const clang::VarDecl>());
    }

    return internal_gen.str();
}

auto CFGCodeGen::ispcCodeGen(CFG::KernelFuncNode* kernel) -> std::string {
    OStreamTy kernel_gen;
    CFG::CFGNode* curr_node = kernel;
    while (curr_node->getNodeType() != CFG::CFGNode::Exit) {
        SPMDFY_INFO("Current Internal node: {}", curr_node->getName());
        if(curr_node->getNodeType() == CFG::CFGNode::Internal)
            ispcCodeGen(dynamic_cast<CFG::InternalNode*>(curr_node));
        curr_node = curr_node->getNext();
    }
    return kernel_gen.str();
}

// CodeGen Visitors

#define DEF_VISITOR(NODE, BASE, NAME)                                          \
    auto CFGCodeGen::Visit##NODE##BASE(const clang::NODE##BASE *NAME)     \
        ->std::string

#define DECL_DEF_VISITOR(NODE, NAME) DEF_VISITOR(NODE, Decl, NAME)

DECL_DEF_VISITOR(Var, var_decl){
    SPMDFY_INFO("Codegen Vistiing VarDecl {}", SRCDUMP(var_decl));
    return "";
}

} // namespace codegen
} // namespace spmdfy