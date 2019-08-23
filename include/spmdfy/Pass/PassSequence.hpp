namespace spmdfy {

namespace pass {

#define SEQUENCE_T(...) using pass_sequence_t = std::tuple<__VA_ARGS__>;

/**
 * \ingroup Pass
 * \brief A tuple representing pass sequence that will be executed in the order
 *
 * */

// clang-format off
SEQUENCE_T(
           locate_ast_nodes_pass_t,
           insert_ispc_nodes_pass_t,
           hoist_shmem_nodes_pass_t,
           detect_partial_nodes_pass_t,
           duplicate_partial_nodes_pass_t,
           print_reverse_cfg_pass_t,
           print_cfg_pass_t
)
// clang-format on

} // namespace pass

} // namespace spmdfy