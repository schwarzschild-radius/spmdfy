#include <spmdfy/Generator/CFGGenerator/CFGCodeGen.hpp>

namespace spmdfy {
namespace codegen {

auto CFGCodeGen::get() -> std::string const { 
    SPMDFY_INFO("Generating Code\n");
    // TraverseCFG(m_node);
    return ""; 
}

auto CFGCodeGen::getFrom(CFG::CFGNode *) -> std::string const {
    return "";
}

auto CFGCodeGen::TraverseCFG(CFG::CFGNode *) -> std::string const {
    return "";
}

} // namespace codegen
} // namespace spmdfy