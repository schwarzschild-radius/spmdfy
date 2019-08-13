#ifndef SPMDFY_CFG_HPP
#define SPMDFY_CFG_HPP

#include <tuple>
#include <memory>
#include <array>

namespace spmdfy {

class CFGNode;

class CFGNode {
  public:
    virtual ~CFGNode() = default;
    enum class CFGEdgeTy { Partial, Complete };
    enum class CFGNodeTy { Entry, Exit, Internal, Conditional, Reconv };
    using SPMDEdge = std::tuple<std::unique_ptr<CFGNode>, CFGEdgeTy>;
  protected:
    CFGNodeTy m_node_type;
};

class EntryNode : public CFGNode {
  private:
    SPMDEdge next;
};

class ExitNode : public CFGNode {
  private:
    SPMDEdge prev;
};

class InternalNode : public CFGNode {
  private:
    SPMDEdge next, prev;
};

class ConditionalNode : public CFGNode {
  private:
    std::array<SPMDEdge, 2> next;
    SPMDEdge prev;
};

class ReconvNode : public CFGNode {
  private:
    std::array<SPMDEdge, 2> prev;
    SPMDEdge next;
};

} // namespace spmdfy

#endif