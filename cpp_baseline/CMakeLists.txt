cmake_minimum_required(VERSION 3.14)

# Project name and settings
project(EigenExample LANGUAGES CXX)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include FetchContent module
include(FetchContent)

# Manually specify GLPK paths
find_library(GLPK_LIB glpk REQUIRED)
find_path(GLPK_INCLUDE_DIR glpk.h REQUIRED)

if(NOT GLPK_LIB OR NOT GLPK_INCLUDE_DIR)
    message(FATAL_ERROR "GLPK not found. Please ensure GLPK is installed.")
endif()

include_directories(${GLPK_INCLUDE_DIR})

# Set up Eigen to be fetched and built from a specific commit
FetchContent_Declare(
  eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 5527e78a64773a84c96c035d4c77b68c9db4f8ad # Use the desired commit hash here (Currently unstable cummunity version)
)

# Make Eigen available
FetchContent_MakeAvailable(eigen)

# Add executables
add_executable(main_exec main.cc calc_mheight.cc opt_mheight.cc)
add_executable(test_eigen_exec test_eigen.cc calc_mheight.cc opt_mheight.cc)
add_executable(test_exec test.cc calc_mheight.cc opt_mheight.cc)
add_executable(test_glpk_exec test_glpk.cc)

# Link Eigen include directory for each executable
target_include_directories(main_exec PRIVATE ${eigen_SOURCE_DIR})
target_include_directories(test_eigen_exec PRIVATE ${eigen_SOURCE_DIR})
target_include_directories(test_exec PRIVATE ${eigen_SOURCE_DIR})
target_include_directories(test_glpk_exec PRIVATE ${eigen_SOURCE_DIR})

# Link GLPK library manually
target_link_libraries(main_exec ${GLPK_LIB})
target_link_libraries(test_eigen_exec ${GLPK_LIB})
target_link_libraries(test_exec ${GLPK_LIB})
target_link_libraries(test_glpk_exec ${GLPK_LIB})

# Set output directories for each executable
set_target_properties(main_exec PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
set_target_properties(test_eigen_exec PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
set_target_properties(test_exec PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
set_target_properties(test_glpk_exec PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)