cmake_minimum_required(VERSION 3.7)
project(lu_net)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES main.cpp src/net.cpp include/net.h include/function.h src/function.cpp)
add_executable(lu_net ${SOURCE_FILES})