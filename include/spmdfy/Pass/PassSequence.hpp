namespace spmdfy {

namespace pass {

#define SEQUENCE_T(...) using pass_sequence_t = std::tuple<__VA_ARGS__>;

SEQUENCE_T(print_cfg_pass_t, print_cfg_pass_t)

} // namespace pass

} // namespace spmdfy