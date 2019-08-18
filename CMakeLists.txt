cmake_minimum_required(VERSION 3.5.1)

project(spmdfy LANGUAGES CXX CUDA)

find_package(LLVM REQUIRED 9 HINTS /usr/lib/llvm-9)

if(${LLVM_FOUND})
    message("LLVM FOUND: ${LLVM_DIR}")
else()
    message(SEND_ERROR "LLVM 9 required due to CUDA 10.1 support")
endif()

add_executable(spmdfy src/main.cpp 
                      src/SpmdfyAction.cpp
                      src/SpmdfyStmtVisitor.cpp
                      src/utils.cpp
                      src/Format.cpp
                      src/CUDA2ISPC.cpp
                      src/CommandLineOpts.cpp
                      src/Logger.cpp
                      src/Generator/SimpleGenerator.cpp
                      src/Generator/CFGGenerator/CFGGenerator.cpp
                      src/Generator/CFGGenerator/ConstructCFG.cpp
                      src/Generator/CFGGenerator/CFG.cpp
                      src/Generator/CFGGenerator/CFGCodeGen.cpp
                      src/Pass/PassManager.cpp
                      src/Pass/Passes/PrintCFGPass.cpp)

target_include_directories(spmdfy PRIVATE include
                                          third_party/json/include 
                                          third_party/spdlog/include)

set_target_properties(spmdfy PROPERTIES CXX_STANDARD 17
                                        CXX_EXTENSIONS OFF)

set(CMAKE_CXX_COMPILER ${LLVM_TOOLS_BINARY_DIR}/clang++)
target_include_directories(spmdfy PRIVATE ${LLVM_INCLUDE_DIRS})
target_include_directories(spmdfy PRIVATE ${CLANG_INCLUDE_DIRS})
target_compile_definitions(spmdfy PRIVATE ${LLVM_DEFINITIONS})

if(CMAKE_BUILD_TYPE STREQUAL Debug)
    message("Compiling in debug mode, Logging enabled")
    target_compile_definitions(spmdfy PRIVATE SPMDFY_DEBUG)
endif()

set(CLANG_LIBS clangFrontend
               clangDriver
               clangSerialization
               clangParse
               clangSema
               clangAnalysis
               clangAST
               clangBasic
               clangEdit
               clangTooling
               clangLex
               clangASTMatchers
               clangRewrite
               clangFormat
               clangApplyReplacements
               clangToolingCore
               clangToolingInclusions)

set(LLVM_LIBS LLVM)

target_link_directories(spmdfy PRIVATE ${LLVM_LIBRARY_DIRS})
target_link_libraries(spmdfy PRIVATE ${CLANG_LIBS} ${LLVM_LIBS})

# Tests and CTest
enable_testing()
add_subdirectory(examples)
add_test(Test_Transpose examples/transpose/transpose)
add_test(Test_saxpy examples/saxpy/saxpy)
add_test(Test_Shared_Memory examples/CUDA_Features/Shared_Memory/shared_memory)
add_test(Test_Atomic examples/CUDA_Features/Atomic/atomic)