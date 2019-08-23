//===- CFG.hpp - Classes for representing CFG Nodes -----------*- C++ -*-===//
//
//===----------------------------------------------------------------------===//
//
//  This file defines the CFG subclasses.
//
//===----------------------------------------------------------------------===//

#ifndef SPMDFY_CFG_HPP
#define SPMDFY_CFG_HPP

#include <clang/AST/RecursiveASTVisitor.h>

#include <spmdfy/Logger.hpp>
#include <spmdfy/utils.hpp>

#include <array>
#include <cassert>
#include <memory>
#include <tuple>
#include <variant>

namespace spmdfy {

namespace cfg {

/// Variant type using to represent any internal node
using InternalNodeTy = std::variant<const clang::Decl *, const clang::Stmt *,
                                    const clang::Expr *, const clang::Type *>;

class CFGNode;
class BiDirectNode;
class ExitNode;

/**
 * \class CFGEdge
 * \ingroup CFG
 *
 * \brief Represents edges in the CFG
 * This class provides interface to query edges in the CFG. In Graph theory, an
 * edge is tuple of vertices e(v, u), where e is a directed edge from v -> u. In
 * CFGEdge class, The owner of the edge is always the starting node pointing to
 * some terminal node.
 *
 * */
class CFGEdge {
  public:
    enum Edge { Partial, Complete };

    // getters
    auto getEdgeType() -> Edge const { return m_edge; }
    auto getEdgeTypeName() -> std::string const;
    auto getTerminal() -> CFGNode *const;

    // setters
    auto setTerminal(CFGNode *terminal, Edge edge_type = Complete) -> CFGNode *;
    auto setEdgeType(Edge edge_type) -> Edge;

  private:
    Edge m_edge;
    CFGNode *m_terminal;
};

// :CFGEdge

/**
 * \class CFGNoode
 * \ingroup CFG
 *
 * \brief Represents nodes in the CFG
 * This class provides interface to query nodes in the CFG. A CFGNode consists
 * of a node of Node enum type.
 *
 * */
class CFGNode {
  public:
    virtual ~CFGNode() {}
    /// Enumeration representing various types of CFGNodes
    enum Node {
        Forward,
        Backward,
        BiDirect,
        GlobalVar,
        StructDecl,
        KernelFunc,
        DeviceFunc,
        Conditional,
        IfStmt,
        ForStmt,
        Reconv,
        Internal,
        Exit,
        ISPCBlock,
        ISPCBlockExit,
        ISPCGrid,
        ISPCGridExit
    };

    /// Enumeration representing the position of the node in CFG
    enum Context { Global, Kernel, Device };

    // getters
    auto getNodeType() -> Node const { return m_node_type; }
    auto getContextType() -> Context const { return m_context; }
    auto setContext(Context &context) -> bool {
        m_context = context;
        return false;
    }
    auto getNodeTypeName() -> std::string const;
    auto getContextTypeName() -> std::string const;

    // virtual methods
    virtual auto getSource() -> std::string const;
    virtual auto setSource(const std::string &) -> std::string;
    virtual auto getName() -> std::string const;
    virtual auto splitEdge(BiDirectNode *) -> BiDirectNode *;
    virtual auto getNext() -> CFGNode *const;
    virtual auto setNext(CFGNode *node,
                         CFGEdge::Edge edge_type = CFGEdge::Complete)
        -> CFGNode *;
    virtual auto getPrevious() -> CFGNode *const;
    virtual auto setPrevious(CFGNode *node,
                             CFGEdge::Edge edge_type = CFGEdge::Complete)
        -> CFGNode *;

  protected:
    std::string m_source, m_name;
    Context m_context;
    Node m_node_type;
};

// :CFGNode

/**
 * \class GlobalVarNode
 * \ingroup CFG
 *
 * \brief Represents Global variables in the SpmdTranslationUnit
 * References a global variable declaration in the AST.
 *
 * */
class GlobalVarNode : public CFGNode {
  public:
    GlobalVarNode(clang::ASTContext &ast_context,
                  const clang::VarDecl *var_decl);
    /// returns name of the string
    auto getName() -> std::string const override {
        return m_var_decl->getNameAsString();
    }
    /// returns Var as it is the declaration kind of any variables decl;
    auto getDeclKindString() -> std::string const {
        return m_var_decl->getDeclKindName();
    }

  private:
    const clang::VarDecl *m_var_decl;

    // AST context
    clang::ASTContext &m_ast_context;
};

// :GlobalVar

/**
 * \class ForwardNode
 * \ingroup CFG
 *
 * \brief Represents a node that point forward in the CFG.
 * Has only one edge that points forward in the CFG.
 *
 * */
class ForwardNode : public virtual CFGNode {
  public:
    virtual ~ForwardNode() { delete m_next; }
    ForwardNode();

    // override
    /// gets the next CFGNode in the control flow
    auto getNext() -> CFGNode *const override;

    /// sets the next CFGNode in the control flow
    auto setNext(CFGNode *, CFGEdge::Edge = CFGEdge::Complete)
        -> CFGNode * override;

    /*
     * \brief splitEdge splits the edge
     * from
     * u -> w
     * to
     * u -> v -> w
     *
     * v must be of a bidirectional node(subject to change)
     *
     * */
    auto splitEdge(BiDirectNode *) -> BiDirectNode * override;

  protected:
    CFGEdge *m_next;
};

// :ForwardNode

/**
 * \class BackwardNode
 * \ingroup CFG
 *
 * \brief Represents a node that point backward in the CFG.
 * Has only one edge that points backward in the CFG.
 *
 * */
class BackwardNode : public virtual CFGNode {
  public:
    virtual ~BackwardNode() = default;
    BackwardNode();
    /// gets the previous node in the CFG
    virtual auto getPrevious() -> CFGNode *const;
    /// sets the previous node in the CFG
    virtual auto setPrevious(CFGNode *node,
                             CFGEdge::Edge edge_type = CFGEdge::Complete)
        -> CFGNode *;

  protected:
    CFGEdge *m_prev;
};

// :BackwardNode
/**
 * \class BiDrectNode
 * \ingroup CFG
 *
 * \brief Represents a birectional node in the CFG, a node that moves forward
 * and backward. It combines the functionality of ForwardNode and BackwardNode.
 * Most nodes in the CFG are birectional Nodes
 *
 * */
class BiDirectNode : public ForwardNode, public BackwardNode {
  public:
    virtual ~BiDirectNode() = default;

    BiDirectNode() {
        m_node_type = BiDirect;
        m_name = getNodeTypeName();
    }
};

// :BiDirect Node

/**
 * \class KernelFuncNode
 * \ingroup CFG
 *
 * \brief Holds a entry point into a Kernel Function. It only point forward in
 * the control and also point to the exit node for reverse CFG traversal.
 *
 * */
class KernelFuncNode : public ForwardNode {
  public:
    KernelFuncNode(clang::ASTContext &ast_context,
                   const clang::FunctionDecl *func_decl);

    auto getName() -> std::string const override;
    auto getKernelNode() -> const clang::FunctionDecl *const;
    /// gets the exit node
    auto getExit() -> ExitNode *const;
    /// sets the exit node
    auto setExit(ExitNode *, CFGEdge::Edge edge_type = CFGEdge::Complete)
        -> ExitNode *;

  private:
    const clang::FunctionDecl *m_func_decl;
    CFGEdge *m_exit;

    // AST context
    clang::ASTContext &m_ast_context;
};

// :KernelFuncNode
/**
 * \class ConditionalNode
 * \ingroup CFG
 *
 * \brief Represents a conditional control-flow in the CFG. It models an If-Then
 * statement. It contains an edge to the special reconvergence node. The place
 * where a divergent control flow meets.
 *
 * */
class ConditionalNode : public BiDirectNode {
  public:
    virtual ~ConditionalNode() { delete reconv; }
    ConditionalNode(clang::ASTContext &ast_context, const clang::Stmt *stmt);
    /// gets the reconvergence point
    auto getReconv() -> CFGNode *const;
    /// sets the reconvergence point
    auto setReconv(CFGNode *node, CFGEdge::Edge edge_type = CFGEdge::Complete)
        -> CFGNode *;

  protected:
    clang::ASTContext &m_ast_context;
    const clang::Stmt *m_cond_stmt;
    CFGEdge *true_b, *reconv;
};

// :ConditionalNode
/**
 * \class IfStmtNode
 * \ingroup CFG
 *
 * \brief Represents a conditional control flow of an if statement in the CFG.
 * It models a if-then-else statemenet.
 *
 * */
class IfStmtNode : public ConditionalNode {
  public:
    ~IfStmtNode() { delete false_b; }
    IfStmtNode(clang::ASTContext &ast_context, const clang::IfStmt *if_stmt);

    /// gets the If statement's AST node
    auto getIfStmt() -> const clang::IfStmt *const {
        return llvm::cast<const clang::IfStmt>(m_cond_stmt);
    }

    /// calls split edge on the true edge and this the default behavious when
    /// splitEdge is called
    auto splitTrueEdge(BiDirectNode *) -> BiDirectNode *;

    /// calls split edge on the false edge
    auto splitFalseEdge(BiDirectNode *) -> BiDirectNode *;

    /// returns the pointer to the true block
    auto getTrueBlock() -> CFGNode *const;

    /// returns the pointer to the false block
    auto getFalseBlock() -> CFGNode *const;

    /// sets the pointer to the true block
    auto setTrueBlock(CFGNode *node,
                      CFGEdge::Edge edge_type = CFGEdge::Complete) -> CFGNode *;

    /// sets the pointer to the false block
    auto setFalseBlock(CFGNode *node,
                       CFGEdge::Edge edge_type = CFGEdge::Complete)
        -> CFGNode *;

  private:
    CFGEdge *false_b;
};

// :IfStmt

/**
 * \class ForStmtNode
 * \ingroup CFG
 *
 * \brief Represents a for control-flow statement in CFG. It models an if-then
 * statement in the CFG.
 *
 * */

class ForStmtNode : public ConditionalNode {
  public:
    ~ForStmtNode() = default;
    ForStmtNode(clang::ASTContext &ast_context, const clang::ForStmt *for_stmt);
    // getters
    /// gets the For statement's AST node
    auto getForStmt() -> const clang::ForStmt *const {
        return llvm::cast<const clang::ForStmt>(m_cond_stmt);
    }
};

// :ForStmtNode

/**
 * \class ReconvNode
 * \ingroup CFG
 *
 * \brief Special Node to represent a reconvering control flow in the CFG.
 * Any control flow after the node is a convergenet control flow as SIMD
 * guarantees maximal convergence
 *
 * */
class ReconvNode : public BiDirectNode {
  public:
    ~ReconvNode() = default;
    ReconvNode(ConditionalNode *cond_node);

    // getters
    /// returns null as the I don't find a reason to traverse back through the
    /// reconv node(subject to change)
    auto setPrevious(CFGNode *node, CFGEdge::Edge edge_type)
        -> CFGNode * override;

    /// sets the pointer to the start of the conditional control flow statement
    auto setBack(CFGNode *node, CFGEdge::Edge edge_type = CFGEdge::Complete)
        -> CFGNode *;

    /// returns the pointer to the start of the conditional control flow
    /// statement
    auto getBack() -> CFGNode *const;

  private:
    CFGEdge *back;
};

// :ReconvNode
/**
 * \class InternalNode
 * \ingroup CFG
 *
 * \brief Represent any no control flow node in the CFG. It also points to any
 * node in the AST. It allows for extensibility of CFG.
 *
 * */
class InternalNode : public BiDirectNode {
  public:
    ~InternalNode() = default;
    InternalNode(clang::ASTContext &ast_context, InternalNodeTy node);

    /// returns the source of the AST Node
    auto getSource() -> std::string const override;

    /// returns the name of internal(name, source, source, typename) Node
    auto getInternalNodeName() -> std::string const;

    /// returns the variant of the internal node
    auto getInternalNode() -> InternalNodeTy const;

    /// returns the node as specific type casted node in the AST
    template <typename ASTNodeTy> auto getInternalNodeAs() -> ASTNodeTy * {
        return std::visit(
            Overload{[](const clang::Decl *decl) {
                         return reinterpret_cast<ASTNodeTy *>(decl);
                     },
                     [](const clang::Stmt *stmt) {
                         return reinterpret_cast<ASTNodeTy *>(stmt);
                     },
                     [](const clang::Expr *expr) {
                         return reinterpret_cast<ASTNodeTy *>(expr);
                     },
                     [](const clang::Type *type) {
                         return reinterpret_cast<ASTNodeTy *>(type);
                     }},
            m_node);
    }

  private:
    InternalNodeTy m_node;

    // AST context
    clang::ASTContext &m_ast_context;
};

// :InternalNode
/**
 * \class ExitNode
 * \ingroup CFG
 *
 * \brief Represents the stop of a contro flow of a function.
 *
 * */
class ExitNode : public BackwardNode {
  public:
    ~ExitNode() = default;
    ExitNode();
};

// :ExitNode
/**
 * \class ISPCBlockNode
 * \ingroup CFG
 *
 * \brief Represents ISPC block start node
 *
 * */
class ISPCBlockNode : public BiDirectNode {
  public:
    ~ISPCBlockNode() = default;
    ISPCBlockNode();

  private:
};

// :ISPCBlockNode
/**
 * \class ISPCBlockExitNode
 * \ingroup CFG
 *
 * \brief Represents ISPC block end node
 *
 * */
class ISPCBlockExitNode : public BiDirectNode {
  public:
    ~ISPCBlockExitNode() = default;
    ISPCBlockExitNode();
};

// :ISPCBlockExitNode
/**
 * \class ISPCGridNode
 * \ingroup CFG
 *
 * \brief Represents ISPC grid start node
 *
 * */
class ISPCGridNode : public BiDirectNode {
  public:
    ~ISPCGridNode() = default;
    ISPCGridNode();
};

// :ISPCGridNode
/**
 * \class ISPCGridExitNode
 * \ingroup CFG
 *
 * \brief Represents ISPC grid end node
 *
 * */
class ISPCGridExitNode : public BiDirectNode {
  public:
    ~ISPCGridExitNode() = default;
    ISPCGridExitNode();
};

// :ISPCGridExitNode

/// removes a CFGNode from the control flow
/// @params node to be removed
/// @return node that was removed
auto rmCFGNode(CFGNode *node) -> cfg::CFGNode *;

// :utils

} // namespace cfg

} // namespace spmdfy

#endif