#include <spmdfy/Pass/PassManager.hpp>

namespace spmdfy {

namespace pass {

auto PassManager::runPassSequence() -> bool {
    // clang-format off
    SPMDFY_INFO("===============================Running through pass sequence===============================");
    // clang-format on

    bool return_value = std::apply(
        [&](auto &... pass) -> bool { return (!pass.invoke() && ...); },
        pass_sequence);

    // clang-format off
    SPMDFY_INFO("===================================End of pass sequennce===================================");
    // clang-format on
    return return_value;
}

auto PassManager::initPassSequence() -> void {
    // clang-format off
    SPMDFY_INFO("===================================Initalizing Pass sequence===============================");
    // clang-format on
    std::apply(
        [&](auto &... pass) -> void { (pass.set_opts(m_spmd_tutbl), ...); },
        pass_sequence);
}

} // namespace pass
} // namespace spmdfy