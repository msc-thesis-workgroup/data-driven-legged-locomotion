cmake_minimum_required(VERSION 3.16)

# INTERPROCEDURAL_OPTIMIZATION is enforced when enabled.
set(CMAKE_POLICY_DEFAULT_CMP0069 NEW)
# Default to GLVND if available.
set(CMAKE_POLICY_DEFAULT_CMP0072 NEW)

# This line has to appear before 'PROJECT' in order to be able to disable incremental linking
set(MSVC_INCREMENTAL_DEFAULT ON)

project(
  data_driven_legged_locomotion
  VERSION 0.1.0
  DESCRIPTION "Crowdsourcing algorithms for legged locomotion"
)

include(FetchContent)

FetchContent_Declare(
  mujoco_mpc
  GIT_REPOSITORY https://github.com/AntoSave/mujoco_mpc.git
  GIT_TAG main
)

FetchContent_MakeAvailable(mujoco_mpc)