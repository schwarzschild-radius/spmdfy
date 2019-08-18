#ifndef PASS_HANDLER_HPP
#define PASS_HANDLER_HPP

#include <spmdfy/Logger.hpp>
#include <spmdfy/utils.hpp>
#include <string>
#include <utility>
#include <vector>

#include <clang/AST/RecursiveASTVisitor.h>
#include <spmdfy/CFG/CFG.hpp>
#include <spmdfy/Pass/PassWorkspace.hpp>

namespace spmdfy {

namespace pass {

using SpmdTUTy = std::vector<cfg::CFGNode *>;

#define PASS(FUNCTION, PASS_TYPE_NAME)                                         \
    struct PASS_TYPE_NAME {                                                    \
        constexpr static auto name{#PASS_TYPE_NAME};                           \
        decltype(make_handler(name, FUNCTION)) handler{name, FUNCTION};        \
        void set_opts(SpmdTUTy &spmd_tutbl, clang::ASTContext &ast_context,    \
                      Workspace &workspace) {                                  \
            handler.set_opts(spmd_tutbl, ast_context, workspace);              \
        }                                                                      \
        bool invoke() { return handler.invoke(); }                             \
    };

template <class Fn, class Tuple, unsigned long... I>
bool invoke_impl(Fn &&function, SpmdTUTy &spmd_tu,
                 clang::ASTContext &ast_context, Workspace &workspace,
                 const Tuple &invoke_arguments, std::index_sequence<I...>) {
    return function(spmd_tu, ast_context, workspace,
                    std::get<I>(invoke_arguments)...);
}

template <typename ReturnTy, typename... ParamTy> struct PassHandler {
    std::string name = "";
    std::tuple<ParamTy...> invoke_arguments;
    SpmdTUTy *m_spmd_tutbl;
    clang::ASTContext *m_ast_context;
    Workspace *m_workspace;

    ReturnTy (*function)(SpmdTUTy &, clang::ASTContext &, Workspace &,
                         ParamTy...);

    PassHandler(const char *name,
                ReturnTy (*param_function)(SpmdTUTy &, clang::ASTContext &,
                                           Workspace &, ParamTy...))
        : name(name), function(param_function) {}

    void set_opts(SpmdTUTy &spmd_tutbl, clang::ASTContext &ast_context,
                  Workspace &workspace) {
        m_spmd_tutbl = &spmd_tutbl;
        m_ast_context = &ast_context;
        m_workspace = &workspace;
    }

    bool invoke() {
        SPMDFY_INFO("Invoking pass {}", name);
        return invoke_impl(
            function, *m_spmd_tutbl, *m_ast_context, *m_workspace,
            invoke_arguments,
            std::make_index_sequence<
                std::tuple_size<std::tuple<ParamTy...>>::value>());
    }
};

template <typename R, typename... T>
static PassHandler<R, T...> make_handler(const char *name,
                                         R(f)(SpmdTUTy &, clang::ASTContext &,
                                              Workspace &, T...)) {
    return PassHandler<R, T...>{name, f};
}

} // namespace pass

} // namespace spmdfy

#endif