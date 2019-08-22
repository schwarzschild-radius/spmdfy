#ifndef PRINT_REVERSE_CFG_PASS_HPP
#define PRINT_REVERSE_CFG_PASS_HPP

#include <spmdfy/Pass/PassHandler.hpp>

namespace spmdfy {

namespace pass {

bool print_reverse_cfg_pass(SpmdTUTy&, clang::ASTContext&, Workspace&);

PASS(print_reverse_cfg_pass, print_reverse_cfg_pass_t);

} // namespace pass
} // namespace spmdfy

#endif