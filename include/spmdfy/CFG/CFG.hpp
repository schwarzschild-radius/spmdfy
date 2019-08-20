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

using InternalNodeTy = std::variant<const clang::Decl *, const clang::Stmt *,
                                    const clang::Expr *, const clang::Type *>;

class CFGNode;
class BiDirectNode;

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

class CFGNode {
  public:
    virtual ~CFGNode() {}

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

class GlobalVarNode : public CFGNode {
  public:
    GlobalVarNode(clang::ASTContext &ast_context,
                  const clang::VarDecl *var_decl)
        : m_ast_context(ast_context) {
        m_var_decl = var_decl;
        m_name = var_decl->getNameAsString();
        m_node_type = GlobalVar;
        m_context = Global;
        SPMDFY_INFO("Creating GlobalVarNode {}", m_name);
    }

    auto getName() -> std::string const override {
        return m_var_decl->getNameAsString();
    }

    auto getDeclKindString() -> std::string const {
        return m_var_decl->getDeclKindName();
    }

  private:
    const clang::VarDecl *m_var_decl;

    // AST context
    clang::ASTContext &m_ast_context;
};

// :GlobalVar

class ForwardNode : public virtual CFGNode {
  public:
    virtual ~ForwardNode() { delete m_next; }
    ForwardNode() {
        m_node_type = Forward;
        m_name = getNodeTypeName();
        m_next = new CFGEdge();
    }

    // override
    auto getNext() -> CFGNode *const override;
    auto setNext(CFGNode *, CFGEdge::Edge = CFGEdge::Complete)
        -> CFGNode * override;
    auto splitEdge(BiDirectNode *) -> BiDirectNode * override;

  protected:
    CFGEdge *m_next;
};

// :ForwardNode

class BackwardNode : public virtual CFGNode {
  public:
    virtual ~BackwardNode() { delete m_prev; }
    BackwardNode() {
        m_node_type = Backward;
        m_name = getNodeTypeName();
        m_prev = new CFGEdge();
    }

    virtual auto getPrevious() -> CFGNode *const;
    virtual auto setPrevious(CFGNode *node,
                             CFGEdge::Edge edge_type = CFGEdge::Complete)
        -> CFGNode *;

  protected:
    CFGEdge *m_prev;
};

// :BackwardNode

class BiDirectNode : public ForwardNode, public BackwardNode {
  public:
    virtual ~BiDirectNode() = default;

    BiDirectNode() {
        m_node_type = BiDirect;
        m_name = getNodeTypeName();
    }
};

// :BiDirect Node

class KernelFuncNode : public ForwardNode {
  public:
    KernelFuncNode(clang::ASTContext &ast_context,
                   const clang::FunctionDecl *func_decl)
        : m_ast_context(ast_context) {
        SPMDFY_INFO("Creating KerneFuncNode {}", func_decl->getNameAsString());
        m_func_decl = func_decl;
        m_node_type = KernelFunc;
        m_name = getNodeTypeName();
        m_context = Global;
    }

    auto getName() -> std::string const override;
    auto getKernelNode() -> const clang::FunctionDecl *const;

  private:
    const clang::FunctionDecl *m_func_decl;

    // AST context
    clang::ASTContext &m_ast_context;
};

// :KernelFuncNode

class ConditionalNode : public BiDirectNode {
  public:
    virtual ~ConditionalNode() { delete reconv; }
    ConditionalNode(clang::ASTContext &ast_context, const clang::Stmt *stmt)
        : m_ast_context(ast_context) {
        m_node_type = Conditional;
        m_name = getNodeTypeName();
        true_b = m_next;
        reconv = new CFGEdge();
        m_cond_stmt = stmt;
    }

    auto getReconv() -> CFGNode *const;
    auto setReconv(CFGNode *node, CFGEdge::Edge edge_type = CFGEdge::Complete)
        -> CFGNode *;

  protected:
    clang::ASTContext &m_ast_context;
    const clang::Stmt *m_cond_stmt;
    CFGEdge *true_b, *reconv;
};

// :ConditionalNode

class IfStmtNode : public ConditionalNode {
  public:
    ~IfStmtNode() { delete false_b; }
    IfStmtNode(clang::ASTContext &ast_context, const clang::IfStmt *if_stmt)
        : ConditionalNode(ast_context, if_stmt) {
        m_node_type = IfStmt;
        m_name = getNodeTypeName();
        false_b = new CFGEdge();
        m_source = "if (" +
                   sourceDump(ast_context.getSourceManager(),
                              ast_context.getLangOpts(),
                              if_stmt->getCond()->getSourceRange().getBegin(),
                              if_stmt->getCond()->getSourceRange().getEnd()) +
                   ")";
    }

    // getters
    auto getIfStmt() -> const clang::IfStmt *const {
        return llvm::cast<const clang::IfStmt>(m_cond_stmt);
    }
    auto splitTrueEdge(BiDirectNode *) -> BiDirectNode *;
    auto splitFalseEdge(BiDirectNode *) -> BiDirectNode *;
    auto getTrueBlock() -> CFGNode *const;
    auto getFalseBlock() -> CFGNode *const;
    auto setTrueBlock(CFGNode *node,
                      CFGEdge::Edge edge_type = CFGEdge::Complete) -> CFGNode *;
    auto setFalseBlock(CFGNode *node,
                       CFGEdge::Edge edge_type = CFGEdge::Complete)
        -> CFGNode *;

  private:
    CFGEdge *false_b;
};

class ForStmtNode : public ConditionalNode {
  public:
    ~ForStmtNode() = default;
    ForStmtNode(clang::ASTContext &ast_context, const clang::ForStmt *for_stmt)
        : ConditionalNode(ast_context, for_stmt) {
        m_node_type = ForStmt;
        m_source = sourceDump(ast_context.getSourceManager(),
                              ast_context.getLangOpts(),
                              for_stmt->getSourceRange().getBegin(),
                              for_stmt->getBody()->getSourceRange().getBegin());
    }
    // getters
    auto getForStmt() -> const clang::ForStmt *const {
        return llvm::cast<const clang::ForStmt>(m_cond_stmt);
    }
};

// :ForStmtNode

class ReconvNode : public BiDirectNode {
  public:
    ~ReconvNode() = default;
    ReconvNode(ConditionalNode *cond_node) {
        m_node_type = Reconv;
        m_name = "ReconvNode";
        m_source = std::string();
        m_context = Kernel;
        back = m_prev;
        back->setTerminal(cond_node);
    }

    // getters
    auto setPrevious(CFGNode *node, CFGEdge::Edge edge_type)
        -> CFGNode * override;
    auto setBack(CFGNode *node, CFGEdge::Edge edge_type = CFGEdge::Complete)
        -> CFGNode *;
    auto getBack() -> CFGNode *const;

  private:
    CFGEdge *back;
};

// :ReconvNode

class InternalNode : public BiDirectNode {
  public:
    ~InternalNode() = default;
    InternalNode(clang::ASTContext &ast_context, InternalNodeTy node)
        : m_node(node), m_ast_context(ast_context) {
        SPMDFY_INFO("Creating InternalNode {}", getInternalNodeName());
        m_node_type = Internal;
        m_name = getInternalNodeName();
    }

    // override
    auto getSource() -> std::string const override;

    // getters
    auto getInternalNodeName() -> std::string const;

    auto getInternalNode() -> InternalNodeTy const;

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

class ExitNode : public BackwardNode {
  public:
    ~ExitNode() = default;
    ExitNode() {
        SPMDFY_INFO("Creating ExitNode");
        m_node_type = Exit;
        m_name = "ExitNode";
        m_source = std::string();
    }
};

// :ExitNode

class ISPCBlockNode : public BiDirectNode {
  public:
    ~ISPCBlockNode() = default;
    ISPCBlockNode() {
        SPMDFY_INFO("Creating ISPCBlockNode");
        m_node_type = ISPCBlock;
    }

  private:
};

// :ISPCBlockNode

class ISPCBlockExitNode : public BiDirectNode {
  public:
    ~ISPCBlockExitNode() = default;
    ISPCBlockExitNode() {
        SPMDFY_INFO("Creating ISPCBlockExitNode");
        m_node_type = ISPCBlockExit;
        m_source = "ISPC_BLOCK_END";
    }
};

// :ISPCBlockExitNode

class ISPCGridNode : public BiDirectNode {
  public:
    ~ISPCGridNode() = default;
    ISPCGridNode() {
        SPMDFY_INFO("Creating ISPCGridNode");
        m_node_type = ISPCGrid;
        m_source = "ISPC_GRID_START";
    }
};

// :ISPCGridNode

class ISPCGridExitNode : public BiDirectNode {
  public:
    ~ISPCGridExitNode() = default;
    ISPCGridExitNode() {
        SPMDFY_INFO("Creating ISPCGridExitNode");
        m_node_type = ISPCGridExit;
        m_source = "ISPC_GRID_END";
    }
};

// :ISPCGridExitNode

} // namespace cfg

} // namespace spmdfy

#endif