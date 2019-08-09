include(${CMAKE_SOURCE_DIR}/cmake/FindISPC.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/FindSPMDfy.cmake)

add_spmdfy_source(transpose_ispc_target transpose.cu transpose.ispc HINTS ${CMAKE_BINARY_DIR}
                  ISPC_DIR ${CMAKE_CURRENT_BINARY_DIR}
                  DUMP_JSON)

add_ispc_library(transpose_ispc ${CMAKE_CURRENT_BINARY_DIR}/transpose.ispc HEADER transpose.h 
                                         HEADER_DIR ${CMAKE_CURRENT_BINARY_DIR})

add_dependencies(transpose_ispc transpose_ispc_target)
enable_language(CUDA)
add_executable(transpose main.cu transpose.cu)
target_link_libraries(transpose PRIVATE transpose_ispc)
set_target_properties(transpose PROPERTIES LINKER_LANGUAGE CUDA)
target_include_directories(transpose PRIVATE ${transpose_ispc_HEADER_DIR} PRIVATE ${CMAKE_SOURCE_DIR}/examples/utils)