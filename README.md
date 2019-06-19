# spmdfy
A transpiler from CUDA to ISPC using libTooling

## Requirements
1. CUDA 9.0 - 9.2
2. ISPC - use built in alloy.py script to install
3. clang 9 - with libclang and llvm-tools
4. CMake 3.5.0 or greater

## Build Instructions
    mkdir build && cd build
    cmake -G Ninja -DLLVM_DIR=path_to_llvm_cmake_dir ..