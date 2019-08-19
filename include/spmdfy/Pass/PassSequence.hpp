namespace spmdfy {

namespace pass {

#define SEQUENCE_T(...) using pass_sequence_t = std::tuple<__VA_ARGS__>;

// clang-format off
SEQUENCE_T(
           find_syncthreads_nodes_pass_t,
           insert_ispc_nodes_pass_t,
           print_cfg_pass_t
)
// clang-format on

} // namespace pass

} // namespace spmdfy