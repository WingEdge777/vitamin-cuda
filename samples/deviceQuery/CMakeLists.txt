cmake_minimum_required(VERSION 3.18) # CUDA 架构支持需要较新的 CMake 版本
project(DeviceQuery LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# "native" for current arch
# "all" or specific "8.0;8.6;8.9"
if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "native")
endif()

find_package(CUDAToolkit REQUIRED)

add_executable(deviceQuery deviceQuery.cpp)


if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(deviceQuery PRIVATE 
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>
        $<$<COMPILE_LANGUAGE:CXX>:-O3>
    )
endif()


target_link_libraries(deviceQuery PRIVATE CUDA::cudart)

message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")