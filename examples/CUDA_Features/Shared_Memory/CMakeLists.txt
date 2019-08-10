include(${CMAKE_SOURCE_DIR}/cmake/FindISPC.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/FindSPMDfy.cmake)

add_spmdfy_source(shared_memory_ispc_target shared_memory.cu shared_memory.ispc HINTS ${CMAKE_BINARY_DIR}
                  ISPC_DIR ${CMAKE_CURRENT_BINARY_DIR})

add_ispc_library(shared_memory_ispc ${CMAKE_CURRENT_BINARY_DIR}/shared_memory.ispc HEADER shared_memory.h 
                                         HEADER_DIR ${CMAKE_CURRENT_BINARY_DIR})

add_dependencies(shared_memory_ispc shared_memory_ispc_target)
enable_language(CUDA)
add_executable(shared_memory main.cu shared_memory.cu)
target_link_libraries(shared_memory PRIVATE shared_memory_ispc)
set_target_properties(shared_memory PROPERTIES LINKER_LANGUAGE CUDA)
target_include_directories(shared_memory PRIVATE ${shared_memory_ispc_HEADER_DIR} PRIVATE ${CMAKE_SOURCE_DIR}/examples/utils)