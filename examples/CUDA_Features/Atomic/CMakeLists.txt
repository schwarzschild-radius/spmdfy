include(${CMAKE_SOURCE_DIR}/cmake/FindISPC.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/FindSPMDfy.cmake)

add_spmdfy_source(atomic_ispc_target atomic.cu atomic.ispc HINTS ${CMAKE_BINARY_DIR}
                  ISPC_DIR ${CMAKE_CURRENT_BINARY_DIR}
                  DUMP_JSON)

add_ispc_library(atomic_ispc ${CMAKE_CURRENT_BINARY_DIR}/atomic.ispc HEADER atomic.h 
                                         HEADER_DIR ${CMAKE_CURRENT_BINARY_DIR})

add_dependencies(atomic_ispc atomic_ispc_target)
enable_language(CUDA)
add_executable(atomic main.cu atomic.cu)
target_link_libraries(atomic PRIVATE atomic_ispc)
set_target_properties(atomic PROPERTIES LINKER_LANGUAGE CUDA)
target_include_directories(atomic PRIVATE ${atomic_ispc_HEADER_DIR} PRIVATE ${CMAKE_SOURCE_DIR}/examples/utils)