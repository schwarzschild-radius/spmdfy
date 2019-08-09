#include <clang/AST/DeclVisitor.h>
#include <clang/AST/RecursiveASTVisitor.h>
#include <clang/AST/StmtVisitor.h>

#include <spmdfy/CUDA2ISPC.hpp>
#include <spmdfy/Generator/Generator.hpp>
#include <spmdfy/utils.hpp>
#include <spmdfy/Logger.hpp>

#include <sstream>

namespace spmdfy {

extern std::string ispc_macros;

template <typename NodeTy>
bool isExpansionInMainFile(clang::SourceManager &sm, NodeTy *node) {
    return sm.isInMainFile(sm.getExpansionLoc(node->getBeginLoc()));
}

class ISPCKernelGenerator
    : public clang::DeclVisitor<ISPCKernelGenerator, std::string>,
      public clang::StmtVisitor<ISPCKernelGenerator, std::string> {
  public:
  private:
};

class SimpleGenerator
    : public ISPCGenerator,
      public clang::ConstDeclVisitor<SimpleGenerator, std::string>,
      public clang::ConstStmtVisitor<SimpleGenerator, std::string> {
    using clang::ConstDeclVisitor<SimpleGenerator, std::string>::Visit;
    using clang::ConstStmtVisitor<SimpleGenerator, std::string>::Visit;

  public:
    SimpleGenerator(clang::ASTContext &context, std::ostringstream &file_writer)
        : m_context(context), m_sm(context.getSourceManager()),
          m_lang_opts(context.getLangOpts()), m_file_writer(file_writer) {
        m_lang_opts.CPlusPlus = true;
        m_lang_opts.Bool = true;
    }

#define DECL_VISITOR(NODE)                                                     \
    auto Visit##NODE##Decl(const clang::NODE##Decl *)->std::string
#define STMT_VISITOR(NODE)                                                     \
    auto Visit##NODE##Stmt(const clang::NODE##Stmt *)->std::string
#define EXPR_VISITOR(NODE)                                                     \
    auto Visit##NODE##Expr(const clang::NODE##Expr *)->std::string
#define CLEAR_STREAM(stream)                                                   \
    stream.str("");                                                            \
    stream.clear();
    DECL_VISITOR(Var);
    DECL_VISITOR(Function);
    DECL_VISITOR(CXXRecord);
    DECL_VISITOR(Namespace);
    DECL_VISITOR(ParmVar);
    DECL_VISITOR(Declarator);

    STMT_VISITOR(Compound);
    STMT_VISITOR(Decl);
    STMT_VISITOR(For);
    STMT_VISITOR(If);

    EXPR_VISITOR(Call);

    auto VisitBinaryOperator(const clang::BinaryOperator *) -> std::string;

    auto handleTranslationUnit(clang::ASTContext &context) -> bool override {
        clang::DeclContext *traverse_decl = llvm::dyn_cast<clang::DeclContext>(
            context.getTranslationUnitDecl());
        if (!traverse_decl) {
            llvm::errs() << "Cannot get DeclContext\n";
            return true;
        }
        m_file_writer << ispc_macros << '\n';
        for (auto D : traverse_decl->decls()) {
            if (!isExpansionInMainFile(m_sm, D)) {
                continue;
            }
            switch (D->getKind()) {
            case clang::Decl::Function:
                m_file_writer << VisitFunctionDecl(
                    llvm::cast<const clang::FunctionDecl>(D));
                CLEAR_STREAM(m_shmem);
                CLEAR_STREAM(m_kernel_context);
                m_scope = -1;
                break;
            case clang::Decl::Var:
                m_file_writer
                    << VisitVarDecl(llvm::cast<const clang::VarDecl>(D));
                break;
            case clang::Decl::Namespace:
                VisitNamespaceDecl(llvm::cast<const clang::NamespaceDecl>(D));
                break;
            default:
                break;
            }
        }
        SPMDFY_INFO("Translation Unit:\n {}\n", m_file_writer.str());
        return true;
    }

  private:
    enum class TUContext {
        GLOBAL,
        CUDA_KERNEL,
        CUDA_DEVICE_FUNCTION,
        CXXCONSTRUCTOR,
        STRUCT
    };
    clang::ASTContext &m_context;
    clang::SourceManager &m_sm;
    clang::LangOptions m_lang_opts;
    std::ostringstream &m_file_writer;
    std::ostringstream m_shmem;
    std::ostringstream m_kernel_context;
    TUContext m_tu_context;
    int m_scope = -1;
};
} // namespace spmdfy