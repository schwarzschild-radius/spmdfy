#include <spmdfy/Pass/PassHandler.hpp>

namespace spmdfy {

namespace pass {

bool print_cfg_pass(SpmdTUTy&);

PASS(print_cfg_pass, print_cfg_pass_t);

} // namespace pass
} // namespace spmdfy