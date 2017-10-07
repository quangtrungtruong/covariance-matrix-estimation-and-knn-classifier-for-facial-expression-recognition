cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(fer_rgbd)

find_package(PCL 1.6 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (fer_rgbd fer_rgbd.cpp)
target_link_libraries (fer_rgbd ${PCL_LIBRARIES})