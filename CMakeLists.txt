cmake_minimum_required(VERSION 3.7)
project(lu_net)

set(CMAKE_CXX_STANDARD 11)

# 添加头文件路径
include_directories(include)

# Find required protobuf package
find_package(Protobuf REQUIRED)
include_directories(${PROTOBUF_INCLUDE_DIRS})

set(SOURCE_FILES main.cpp src/net.cpp src/function.cpp src/io.cpp src/loss_function.cpp proto/lu.pb.cc src/activation_function.cpp)

add_executable(lu_net ${SOURCE_FILES})

target_link_libraries(lu_net ${PROTOBUF_LIBRARIES} gflags glog)