# Copyright (c) Huawei Technologies Co., Ltd. 2019. All rights reserved.

# CMake lowest version requirement
cmake_minimum_required(VERSION 3.5.1)

# project information
project(SubApp)

# Compile options
add_compile_options(-std=c++11)

# Specify target generation path
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "../../../out")
set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall")

# Header path

include_directories(
    /home/ma-user/Ascend/ascend-toolkit/5.0.2.1/acllib/include
    ../inc/
)

# add host lib path

link_directories(
    /home/ma-user/Ascend/ascend-toolkit/5.0.2.1/acllib/lib64/stub/
)

add_executable(main main.cpp utils.cpp)

if (${CMAKE_HOST_SYSTEM_NAME} MATCHES "Windows")
    target_link_libraries(main
        libascendcl)
else ()
    if (${target} MATCHES "Simulator_Function")
        target_link_libraries(main
            funcsim)
    else ()
        target_link_libraries(main
            ascendcl stdc++)
    endif ()
endif ()

install(TARGETS main DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
