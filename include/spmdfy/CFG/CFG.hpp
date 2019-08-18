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

class CFGNode;

class CFGEdge {
  public:
    enum Edge { Partial, Complete };

    // getters
    auto getEdgeType() -> Edge const { return m_edge; }
    auto getEdgeTypeName() -> std::string const;
    auto getTerminal() -> CFGNode *const;

    // setters
    auto setTerminal(CFGNode *terminal, Edge edge_type) -> bool;
    auto setEdgeType(Edge edge_type) -> bool;

  private:
    Edge m_edge;
    CFGNode *m_terminal;
};

class CFGNode {
  public:
    virtual ~CFGNode() = default;
    enum Node {
        GlobalVar,
        StructDecl,
        KernelFunc,
        DeviceFunc,
        IfStmt,
        ForStmt,
        Internal,
        Exit,
        ISPCBlock,
        ISPCBlockExit,
        ISPCGrid,
        ISPCGridExit
    };
    enum Context { Global, Kernel, Device };
    // getters
    virtual auto getNodeType() -> Node const { return m_node_type; }
    virtual auto getContextType() -> Context const { return m_context; }
    auto getNodeTypeName() -> std::string const;
    auto getContextTypeName() -> std::string const;

    // virtual methods
    virtual auto getName() -> std::string const {
        return "No Name";
    }

    virtual auto splitEdge(CFGNode*) -> bool{
        SPMDFY_ERROR("Not supported operation for {}", getNodeTypeName());
        return true;
    }

    virtual auto getNext() -> CFGNode *const {
        SPMDFY_ERROR("Not supported operation for {}", getNodeTypeName());
        return nullptr;
    }
    virtual auto getPrevious() -> CFGNode *const {
        SPMDFY_ERROR("Not supported operation for {}", getNodeTypeName());
        return nullptr;
    }
    virtual auto setNext(CFGNode *node, CFGEdge::Edge edge_type) -> bool {
        SPMDFY_ERROR("Not supported operation for {}", getNodeTypeName());
        return true;
    }
    virtual auto setPrevious(CFGNode *node, CFGEdge::Edge edge_type) -> bool {
        SPMDFY_ERROR("Not supported operation for {}", getNodeTypeName());
        return true;
    }

  protected:
    Context m_context;
    Node m_node_type;
};

class GlobalVarNode : public CFGNode {
  public:
    GlobalVarNode(clang::ASTContext &ast_context,
                  const clang::VarDecl *var_decl)
        : m_ast_context(ast_context) {
        SPMDFY_INFO("Creating GlobalVarNode {}", var_decl->getNameAsString());
        m_var_decl = var_decl;
        m_node_type = GlobalVar;
        m_context = Global;
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

class KernelFuncNode : public CFGNode {
  public:
    KernelFuncNode(clang::ASTContext &ast_context,
                   const clang::FunctionDecl *func_decl)
        : m_ast_context(ast_context) {
        SPMDFY_INFO("Creating KerneFuncNode {}", func_decl->getNameAsString());
        m_func_decl = func_decl;
        m_node_type = KernelFunc;
        m_context = Global;
        next = new CFGEdge();
    }

    auto splitEdge(CFGNode*) -> bool override;
    auto setNext(CFGNode *node, CFGEdge::Edge edge_type) -> bool override;
    auto getNext() -> CFGNode *const override;
    auto getName() -> std::string const override{
        return m_func_decl->getNameAsString();
    }
    auto getKernelNode() -> const clang::FunctionDecl * const{
        return m_func_decl;
    }
    auto getDeclKindString() -> std::string const;

  private:
    const clang::FunctionDecl *m_func_decl;
    CFGEdge *next;

    // AST context
    clang::ASTContext &m_ast_context;
};

class InternalNode : public CFGNode {
  public:
    using NodeTy = std::variant<const clang::Decl *, const clang::Stmt *,
                                const clang::Expr *, const clang::Type *>;

    InternalNode(clang::ASTContext &ast_context, NodeTy node)
        : m_node(node), m_ast_context(ast_context) {
        SPMDFY_INFO("Creating InternalNode {}", getInternalNodeName());
        m_node_type = Internal;
        next = new CFGEdge();
        prev = new CFGEdge();
    }

    // getters
    auto getInternalNodeName() -> std::string const;
    template<typename ASTNodeTy>
    auto getInternalNodeAs() -> ASTNodeTy*{
        return std::visit(visitor{
            [](const clang::Decl *decl) { return reinterpret_cast<ASTNodeTy*>(decl); },
            [](const clang::Stmt *stmt) { return reinterpret_cast<ASTNodeTy*>(stmt); },
            [](const clang::Expr *expr) { return reinterpret_cast<ASTNodeTy*>(expr); },
            [](const clang::Type *type) { return reinterpret_cast<ASTNodeTy*>(type); }}
        , m_node);
    }

    // override
    auto splitEdge(CFGNode*) -> bool override;
    auto getName() -> std::string const override;
    auto getNext() -> CFGNode *const override;
    auto getPrevious() -> CFGNode *const override;
    auto setNext(CFGNode *node, CFGEdge::Edge edge_type) -> bool override;
    auto setPrevious(CFGNode *node, CFGEdge::Edge edge_type) -> bool override;

  private:
    NodeTy m_node;
    CFGEdge *next, *prev;

    // AST context
    clang::ASTContext &m_ast_context;
};

class ExitNode : public CFGNode {
  public:
    ExitNode(){
        SPMDFY_INFO("Creating ExitNode");
        m_node_type = Exit;
        prev = new CFGEdge();
    }
    auto getPrevious() -> CFGNode *const override;
    auto setPrevious(CFGNode *node, CFGEdge::Edge edge_type) -> bool override;
    auto getName() -> std::string const override {
        return "Exit";
    }

  private:
    CFGEdge *prev;
};

class ISPCBlockNode: public CFGNode{
  public:
    ISPCBlockNode(){
        SPMDFY_INFO("Creating ISPCBlockNode");
        m_node_type = ISPCBlock;
        m_context = Kernel;
        prev = new CFGEdge();
        next = new CFGEdge();
    }

    // override
    auto getName() -> std::string const override;
    auto getNext() -> CFGNode *const override;
    auto getPrevious() -> CFGNode *const override;
    auto setNext(CFGNode *node, CFGEdge::Edge edge_type) -> bool override;
    auto setPrevious(CFGNode *node, CFGEdge::Edge edge_type) -> bool override;

  private:
    CFGEdge *prev, *next;
};

class ISPCBlockExitNode: public CFGNode{
  public:
    ISPCBlockExitNode(){
        SPMDFY_INFO("Creating ISPCBlockExitNode");
        m_node_type = ISPCBlockExit;
        m_context = Kernel;
        prev = new CFGEdge();
        next = new CFGEdge();
    }

    // override 
    auto getName() -> std::string const override;
    auto getNext() -> CFGNode *const override;
    auto getPrevious() -> CFGNode *const override;
    auto setNext(CFGNode *node, CFGEdge::Edge edge_type) -> bool override;
    auto setPrevious(CFGNode *node, CFGEdge::Edge edge_type) -> bool override;

  private:
    CFGEdge *prev, *next;
};


class ISPCGridNode: public CFGNode{
  public:
    ISPCGridNode(){
        SPMDFY_INFO("Creating ISPCGridNode");
        m_node_type = ISPCGrid;
        m_context = Kernel;
        prev = new CFGEdge();
        next = new CFGEdge();
    }

    // override
    auto getName() -> std::string const override;
    auto getNext() -> CFGNode *const override;
    auto getPrevious() -> CFGNode *const override;
    auto setNext(CFGNode *node, CFGEdge::Edge edge_type) -> bool override;
    auto setPrevious(CFGNode *node, CFGEdge::Edge edge_type) -> bool override;

  private:
    CFGEdge *prev, *next;
};

class ISPCGridExitNode: public CFGNode{
  public:
    ISPCGridExitNode(){
        SPMDFY_INFO("Creating ISPCGridExitNode");
        m_node_type = ISPCGridExit;
        m_context = Kernel;
        prev = new CFGEdge();
        next = new CFGEdge();
    }

    // override 
    auto getName() -> std::string const override;
    auto getNext() -> CFGNode *const override;
    auto getPrevious() -> CFGNode *const override;
    auto setNext(CFGNode *node, CFGEdge::Edge edge_type) -> bool override;
    auto setPrevious(CFGNode *node, CFGEdge::Edge edge_type) -> bool override;

  private:
    CFGEdge *prev, *next;
};

} // namespace CFG

} // namespace spmdfy

#endif