set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} -std=c++14)

add_subdirectory(CUDA_Features)
# add_subdirectory(finite_difference)
# add_subdirectory(saxpy)
add_subdirectory(transpose)