#include <spmdfy/Pass/Passes/PrintCFGPass.hpp>

namespace spmdfy {

namespace pass {

bool print_cfg_pass(SpmdTUTy& spmd_tu) {
    for (auto node : spmd_tu) {
        SPMDFY_INFO("Visiting Node {}", node->getNodeTypeName());
    }
    return false;
}

} // namespace pass

} // namespace spmdfy