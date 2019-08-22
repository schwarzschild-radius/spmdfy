include(${CMAKE_SOURCE_DIR}/cmake/FindISPC.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/FindSPMDfy.cmake)

add_spmdfy_source(reduce_ispc_target reduce.cu reduce.ispc HINTS ${CMAKE_BINARY_DIR}
                  ISPC_DIR ${CMAKE_CURRENT_BINARY_DIR})

add_ispc_library(reduce_ispc ${CMAKE_CURRENT_BINARY_DIR}/reduce.ispc HEADER reduce.h 
                                         HEADER_DIR ${CMAKE_CURRENT_BINARY_DIR})

add_dependencies(reduce_ispc reduce_ispc_target)
enable_language(CUDA)
add_executable(reduce main.cu reduce.cu)
target_link_libraries(reduce PRIVATE reduce_ispc)
set_target_properties(reduce PROPERTIES LINKER_LANGUAGE CUDA)
target_include_directories(reduce PRIVATE ${reduce_ispc_HEADER_DIR} PRIVATE ${CMAKE_SOURCE_DIR}/examples/utils)