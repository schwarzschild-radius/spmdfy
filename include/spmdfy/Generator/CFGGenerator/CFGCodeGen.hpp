#ifndef CFGCODEGEN_HPP
#define CFGCODEGEN_HPP

#include <spmdfy/Generator/CFGGenerator/CFG.hpp>
#include <spmdfy/Logger.hpp>

#include <sstream>

namespace spmdfy {
namespace codegen {

class CFGCodeGen {
  public:
    using OStreamTy = std::ostringstream;
    CFGCodeGen(const std::vector<CFG::CFGNode *> &node) : m_node(node) {}
    auto get() -> std::string const;
    auto getFrom(CFG::CFGNode *) -> std::string const;
    auto TraverseCFG(CFG::CFGNode *) -> std::string const;
  private:
    const std::vector<CFG::CFGNode *> m_node;
};

} // namespace codegen
} // namespace spmdfy

#endif