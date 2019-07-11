cmake_minimum_required(VERSION 3.5.1)

project(spmdfy LANGUAGES CXX)

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
                      src/Format.cpp)
target_include_directories(spmdfy PRIVATE include)
target_include_directories(spmdfy PRIVATE third_party/json/include)
set_target_properties(spmdfy PROPERTIES CXX_STANDARD 17
                                        CXX_EXTENSIONS OFF)

set(CMAKE_CXX_COMPILER ${LLVM_TOOLS_BINARY_DIR}/clang++)
target_include_directories(spmdfy PRIVATE ${LLVM_INCLUDE_DIRS})
target_include_directories(spmdfy PRIVATE ${CLANG_INCLUDE_DIRS})
target_compile_definitions(spmdfy PRIVATE ${LLVM_DEFINITIONS})

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