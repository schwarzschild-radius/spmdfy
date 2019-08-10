include(${CMAKE_SOURCE_DIR}/cmake/FindISPC.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/FindSPMDfy.cmake)

add_spmdfy_source(saxpy_ispc_target saxpy.cu saxpy.ispc HINTS ${CMAKE_BINARY_DIR}
                  ISPC_DIR ${CMAKE_CURRENT_BINARY_DIR})

add_ispc_library(saxpy_ispc ${CMAKE_CURRENT_BINARY_DIR}/saxpy.ispc HEADER saxpy.h 
                                         HEADER_DIR ${CMAKE_CURRENT_BINARY_DIR})

add_dependencies(saxpy_ispc saxpy_ispc_target)
enable_language(CUDA)
add_executable(saxpy main.cu saxpy.cu)
target_link_libraries(saxpy PRIVATE saxpy_ispc)
set_target_properties(saxpy PROPERTIES LINKER_LANGUAGE CUDA)
target_include_directories(saxpy PRIVATE ${saxpy_ispc_HEADER_DIR} PRIVATE ${CMAKE_SOURCE_DIR}/examples/utils)