cmake_minimum_required(VERSION 4.1)
project(HPC LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES 89)

add_subdirectory(MatMul)
add_subdirectory(VectorSum)

message(STATUS "CUDA Toolkit version: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA include directories: ${CUDAToolkit_INCLUDE_DIRS}")
message(STATUS "CUDA libraries: ${CUDAToolkit_LIBRARIES}")
