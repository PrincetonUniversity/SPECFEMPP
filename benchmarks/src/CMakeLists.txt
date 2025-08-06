cmake_minimum_required(VERSION 3.17.5)

if (NOT DEFINED SPECFEM_BENCHMARKS_BUILD_DIR)
    message(STATUS "SPECFEM_BENCHMARKS_BUILD_DIR is not defined, using default: ${CMAKE_CURRENT_SOURCE_DIR}/../build")
    set(SPECFEM_BENCHMARKS_BUILD_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../build)
endif()

# Add the benchmarks build dir to the clean target
set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_CLEAN_FILES
    ${SPECFEM_BENCHMARKS_BUILD_DIR}
)

# 2D
add_subdirectory(dim2/anisotropic-crystal)
add_subdirectory(dim2/homogeneous-medium-flat-topography)
add_subdirectory(dim2/fluid-solid-interface)
add_subdirectory(dim2/fluid-solid-bathymetry)
add_subdirectory(dim2/solid-solid-interface)
add_subdirectory(dim2/Tromp_2005)
add_subdirectory(dim2/poroelastic-semi-infinite)
add_subdirectory(dim2/homogeneous-cosserat)

# 3D
add_subdirectory(dim3/homogeneous_halfspace)
