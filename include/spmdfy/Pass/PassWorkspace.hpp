#ifndef PASS_WORKSPACE_HPP
#define PASS_WORKSPACE_HPP

#include <queue>
#include <spmdfy/CFG/CFG.hpp>

namespace spmdfy {

namespace pass {
/**
 * \class Worksace
 * \ingroup Pass
 *
 * \brief A global workspace structure which acts as a scratch space for passes
 * and for sharing data
 *
 * */
struct Workspace {
    std::map<std::string, std::queue<cfg::InternalNode *>> syncthreads_queue;
    std::map<std::string, std::queue<cfg::InternalNode *>> shmem_queue;
    std::map<std::string, std::map<int, std::vector<cfg::InternalNode *>>>
        partial_nodes;
};

} // namespace pass

} // namespace spmdfy

#endif