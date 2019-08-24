/** \file CFG.hpp
 *  \brief Classes for representing CFG Nodes
 *  This file defines the CFG subclasses which form the nodes of the CFG.
 *
 *  \author Pradeep Kumar  (schwarzschild-radius/@pt_of_no_return)
 *  \bug No know bugs
 *  \defgroup CFG
 * */

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

/// Variant type used to represent any internal node
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
    /// \enum Edge enum class
    enum Edge {
        Partial, ///< Edge of Partial Type
        Complete ///< Edge of Complete Type
    };

    /**
     * \return returns the EdgeType of Edge
     */
    auto getEdgeType() -> Edge const { return m_edge; }

    /**
     * \return returns stringified version of Edge
     */
    auto getEdgeTypeName() -> std::string const;

    /**
     * \return returns the pointer to the terminal node of CFGNode type
     */
    auto getTerminal() -> CFGNode *const;

    /**
     * \param terminal terminal node that the owner of the edge points to
     * \param edge_type typeo of the edge(default: Complete)
     * \return returns the terminal node that was passed
     */
    auto setTerminal(CFGNode *terminal, Edge edge_type = Complete) -> CFGNode *;

    /// set the edgetype
    auto setEdgeType(Edge edge_type) -> Edge;

  private:
    Edge m_edge;
    CFGNode *m_terminal;
};

// :CFGEdge

/**
 * \class CFGNode
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
    /// \enum Node enum class
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

    /// \enum Enumeration representing the position of the node in CFG
    enum Context { Global, Kernel, Device };

    /**
     * \return returns the node type
     */
    auto getNodeType() -> Node const { return m_node_type; }

    /**
     * \return returns context type
     */
    auto getContextType() -> Context const { return m_context; }

    /**
     * \param context of type Context
     * \return returns the terminal node that was passed
     */
    auto setContext(Context &context) -> bool {
        m_context = context;
        return false;
    }

    /**
     * \return returns stringified version of node
     */
    auto getNodeTypeName() -> std::string const;

    /**
     * \return returns stringified version of node
     */
    auto getContextTypeName() -> std::string const;

    // virtual methods
    /**
     * \return returns the source of the node in the AST or the name of the node
     * if it is not part of the AST
     */
    virtual auto getSource() -> std::string const;

    /**
     * \return sets the source of the node
     */
    virtual auto setSource(const std::string &) -> std::string;

    /**
     * \returns string based on the following node types
     *  1. GlobalVar - variable name
     *  2. Forward - returns Forward
     *  3. Backward - returns Backward
     *  4. KernelFunc - returns the function name
     *  5. Conditional - returns Conditional
     *  6. IfStmt - returns IfStmt
     *  7. ForStmt - returns ForStmt
     *  8. Reconv - returns Reconv
     *  9. Internal - returns the kind of the statement e.g. Var, CallExpr etc
     *  10. ExitNode - return Exit
     */
    virtual auto getName() -> std::string const;

    /**
     * \param node - takes BiDirectNode* to insert next to the current node
     * \return returns the inserted node
     */
    virtual auto splitEdge(BiDirectNode *) -> BiDirectNode *;

    /**
     * \return returns next node in the control flow
     */
    virtual auto getNext() -> CFGNode *const;

    /**
     * sets the next node in the control flow
     * \param node - node to be inserted
     * \param edge_type - type of the control flow edge(default: Complete)
     * \return returns stringified version of node
     */
    virtual auto setNext(CFGNode *node,
                         CFGEdge::Edge edge_type = CFGEdge::Complete)
        -> CFGNode *;

    /**
     * \return returns the previous node in the CFG
     */
    virtual auto getPrevious() -> CFGNode *const;

    /**
     * \return sets the previous node in the CFG
     */
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
    /**
     * \return returns name of the string
     */
    auto getName() -> std::string const override {
        return m_var_decl->getNameAsString();
    }
    /**
     * \return returns Var as it is the declaration kind of any variables decl;
     */
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

    /**
     * \return returns the next CFGNode in the control flow
     */
    auto getNext() -> CFGNode *const override;

    /// sets
    /**
     * sets the next node in the control flow
     * \param node - node to be inserted
     * \param edge_type - type of the edge
     * \return returns the next CFGNode in the control flow
     */
    auto setNext(CFGNode *node, CFGEdge::Edge edge_type = CFGEdge::Complete)
        -> CFGNode * override;

    /*
     * \brief splitEdge splits the edge from current node to the next node
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
    /**
     * \return returns the previous node
     */
    virtual auto getPrevious() -> CFGNode *const;

    /**
     * \param node - node to be set
     * \param edge_type - type of the edge(default = Complete)
     * \return sets the previous node
     */
    virtual auto setPrevious(CFGNode *node,
                             CFGEdge::Edge edge_type = CFGEdge::Complete)
        -> CFGNode *;

  protected:
    CFGEdge *m_prev;
};

// :BackwardNode
/**
 * \class BiDirectNode
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

    /**
     * \return returns the name of the function
     */
    auto getName() -> std::string const override;

    /**
     * \return returns the pointer to FunctionDecl node in the AST
     */
    auto getKernelNode() -> const clang::FunctionDecl *const;

    /**
     * \return returns the handle to exit node(endo of function)
     */
    auto getExit() -> ExitNode *const;

    /**
     * \param node - pointer to exit node
     * \param edge_type - type of edge (default = Complete)
     * \return returns the exit node
     */
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

    /**
     * \return returns the reconvergence node
     */
    auto getReconv() -> CFGNode *const;

    /**
     * \param node - reconv node
     * \param node - type of the edge(default = Complete)
     * \return returns the previous node
     */
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

    /**
     * \return gets the If statement's AST node
     */
    auto getIfStmt() -> const clang::IfStmt *const {
        return llvm::cast<const clang::IfStmt>(m_cond_stmt);
    }

    /**
     * calls split edge on the true edge and this the default behavious when
     * splitEdge is called
     * \param node - node to be inserted
     * \return returns the inserted node
     */
    auto splitTrueEdge(BiDirectNode *) -> BiDirectNode *;

    /**
     * calls split edge on the fase edge
     * \param node - node to be inserted
     * \return returns the inserted node
     */
    auto splitFalseEdge(BiDirectNode *) -> BiDirectNode *;

    /// \return returns the pointer to the true block
    auto getTrueBlock() -> CFGNode *const;

    /// \return returns the pointer to the false block
    auto getFalseBlock() -> CFGNode *const;

    /**
     * sets the pointer to the true block
     * \param node - true block
     * \return returns the inserted node
     */
    auto setTrueBlock(CFGNode *node,
                      CFGEdge::Edge edge_type = CFGEdge::Complete) -> CFGNode *;

    /**
     * sets the pointer to the false block
     * \return returns the inserted node
     */
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

    /// \return gets the For statement's AST node
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

    /// \return returns null as the I don't find a reason to traverse back through the
    /// reconv node(subject to change)
    auto setPrevious(CFGNode *node, CFGEdge::Edge edge_type)
        -> CFGNode * override;

    /**
     * Points the conditional node
     * \param node - node in the CFG
     * \param edge_type - type of the edge
     * \return returns the conditional node
     */
    auto setBack(CFGNode *node, CFGEdge::Edge edge_type = CFGEdge::Complete)
        -> CFGNode *;

    /**
     * \return returns the conditional node
     */
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

    /// \return returns the source of the AST Node
    auto getSource() -> std::string const override;

    /// \return returns the name of internal(name, source, source, typename) Node
    auto getInternalNodeName() -> std::string const;

    /// \return returns the variant of the internal node
    auto getInternalNode() -> InternalNodeTy const;

    /// \tparam Type of node in the AST
    /// \return returns the node as specific type casted node in the AST
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
/// \param node to be removed
/// \return node that was removed
auto rmCFGNode(CFGNode *node) -> cfg::CFGNode *;

// :utils

} // namespace cfg

} // namespace spmdfy

#endif