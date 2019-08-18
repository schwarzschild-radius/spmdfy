#ifndef PASS_WORKSPACE_HPP
#define PASS_WORKSPACE_HPP

#include <queue>
#include <spmdfy/CFG/CFG.hpp>

namespace spmdfy {

namespace pass {

#define WORKSPACE(VAR, TYPE) TYPE VAR;

#define WORKSPACES(...)                                                        \
    struct Workspace {                                                         \
        __VA_ARGS__                                                            \
    };

// clang-format off
WORKSPACES(
    WORKSPACE(syncthrds_queue, std::queue<cfg::InternalNode *>)
)
// clang-format on

} // namespace pass

} // namespace spmdfy

#endif