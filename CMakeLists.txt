cmake_minimum_required(VERSION 3.16)

set(CMAKE_C_COMPILER "clang")
set(CMAKE_CXX_COMPILER "clang++")

# INTERPROCEDURAL_OPTIMIZATION is enforced when enabled.
set(CMAKE_POLICY_DEFAULT_CMP0069 NEW)
# Default to GLVND if available.
set(CMAKE_POLICY_DEFAULT_CMP0072 NEW)

# This line has to appear before 'PROJECT' in order to be able to disable incremental linking
set(MSVC_INCREMENTAL_DEFAULT ON)

set(Python3_FIND_VIRTUALENV ONLY)
set(Python3_FIND_STRATEGY LOCATION)

project(
  data_driven_legged_locomotion
  VERSION 0.1.0
  DESCRIPTION "Crowdsourcing algorithms for legged locomotion"
)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
message("Python3_FOUND:${Python3_FOUND}")
message("Python3_VERSION:${Python3_VERSION}")
message("Python3_Development_FOUND:${Python3_Development_FOUND}")
message("Python3_EXECUTABLE:${Python3_EXECUTABLE}")
#TODO CHANGE BUILD DIR FOR MUJOCO_MPC TO AVOID THE LAST ERROR
# include(FetchContent)
# FetchContent_Declare(
#   mujoco_mpc
#   GIT_REPOSITORY https://github.com/AntoSave/mujoco_mpc.git
#   GIT_TAG main
# )
# FetchContent_GetProperties(mujoco_mpc)
# message(STATUS "mujoco_mpc_SOURCE_DIR=${mujoco_mpc_SOURCE_DIR}")
# message(STATUS "mujoco_mpc_BINARY_DIR=${mujoco_mpc_BINARY_DIR}")
# FetchContent_MakeAvailable(mujoco_mpc)
include(ExternalProject)
ExternalProject_Add(
  mujoco_mpc
  GIT_REPOSITORY https://github.com/AntoSave/mujoco_mpc.git
  GIT_TAG main
)

add_custom_command(TARGET mujoco_mpc POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
                mujoco_mpc-prefix/src/mujoco_mpc-build
                mujoco_mpc-prefix/src/mujoco_mpc/build)
execute_process(
  COMMAND ${Python3_EXECUTABLE} setup.py install
  WORKING_DIRECTORY ${mujoco_mpc_SOURCE_DIR}/python
)