cmake_minimum_required(VERSION 3.20)
project(stb)

# Create a library that holds only header files
add_library(stb INTERFACE)

# Define where the header files are
target_include_directories(stb INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})