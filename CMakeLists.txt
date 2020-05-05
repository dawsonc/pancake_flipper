cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

project(pancak_flipper VERSION 0.1.0 LANGUAGES CXX
    DESCRIPTION "Flips a pancake")

set( CMAKE_CXX_FLAGS "-g -O0" )

find_package(drake CONFIG REQUIRED)
find_package(gflags CONFIG REQUIRED)

# end dependencies

add_executable(arm_visualizer arm_visualizer.cc)
target_link_libraries(arm_visualizer drake::drake gflags)
