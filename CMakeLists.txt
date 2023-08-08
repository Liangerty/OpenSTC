# There are several options need to be set here
# 1. "CMAKE_CUDA_ARCHITECTURES", the CUDA architecture should be specified
#     For example, for Nvidia Quadro GP100, the value should be 60, while for GV100, 70
# 2. "MAX_SPEC_NUMBER" if mixture, the number of species should not beyond 55, which is set by allowing the computation of CH4 in Grimech mechanism.
#     If larger mechanism is to be used, the variable should be modified to be larger or equal to the species number.
#     However, the correctness, if the resource is available for the corresponding kernel may need to be tested.

cmake_minimum_required(VERSION 3.23)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 60)
endif()

add_compile_definitions(MAX_SPEC_NUMBER=20)
add_compile_definitions(MAX_REAC_NUMBER=50)
add_compile_definitions(GPU)

project(OpenSTC CUDA CXX)

set(CMAKE_CUDA_STANDARD 20)

add_subdirectory(src)

include_directories(depends)

#list(APPEND src_file
#  BoundCond.h BoundCond.cpp BoundCond.cuh BoundCond.cu
#  ChemData.h ChemData.cpp
#  DParameter.cu DParameter.h Driver.h Driver.cu
#  Element.h Element.cpp
#  Field.h Field.cu
#  Initialize.h Initialize.cu InviscidScheme.cu InviscidScheme.cuh
#  Limiter.cuh
#  main.cu Mesh.h Mesh.cpp
#  Parallel.h Parallel.cpp Parameter.h Parameter.cpp
#  Reconstruction.cu Reconstruction.cuh
#  TemporalScheme.cu TemporalScheme.cuh Thermo.cuh Thermo.cu TimeAdvanceFunc.cu TimeAdvanceFunc.cuh Transport.cuh Transport.cu
#  ViscousScheme.cu ViscousScheme.cuh
#
#  gxl_lib/Array.hpp
#  gxl_lib/MyString.cpp
#  gxl_lib/Time.cpp gxl_lib/Time.h
#)
#
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})
#
#cmake_policy(SET CMP0074 NEW)
#find_package(MPI REQUIRED)
#
#add_executable(OpenSTC ${src_file})
#
#set_target_properties(OpenSTC PROPERTIES
#  CUDA_SEPARABLE_COMPILATION ON)
#
#set_target_properties(OpenSTC
#  PROPERTIES
#  CXX_STANDARD 20 # A newer version CMake 3.25.2 would support this
#  CXX_EXTENSIONS OFF
#  CXX_STANDARD_REQUIRED ON
#  #  "$<$<COMPILE_LANGUAGE:CXX>:${flags_release}>"
#)
#
#target_include_directories(OpenSTC
#  PUBLIC
#  ${CMAKE_SOURCE_DIR}/depends/include/
#)
#
#target_link_directories(OpenSTC
#  PUBLIC#PRIVATE
#  "$<$<CONFIG:Debug>:${CMAKE_SOURCE_DIR}/depends/lib/debug>"
#  "$<$<CONFIG:Release>:${CMAKE_SOURCE_DIR}/depends/lib/release>"
#)
#
##target_compile_options(2_CLionGPU
##  PUBLIC#private
##  ${flags}
##  "$<$<COMPILE_LANGUAGE:CXX>:${flags_release}>"
##  )
#
#target_link_libraries(OpenSTC
#  PUBLIC#PRIVATE
#  "$<$<CONFIG:Debug>:fmtd>"
#  "$<$<CONFIG:Release>:fmt>"
#  MPI::MPI_CXX
#)
