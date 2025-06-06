cmake_minimum_required(VERSION 3.17.5)

if (NOT DEFINED BENCHMARKS_BUILD_DIR)
    set(BENCHMARKS_BUILD_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../build)
endif()

# 2D
add_subdirectory(dim2/anisotropic-crystal)
add_subdirectory(dim2/homogeneous-medium-flat-topography)
add_subdirectory(dim2/fluid-solid-interface)
add_subdirectory(dim2/fluid-solid-bathymetry)
add_subdirectory(dim2/solid-solid-interface)
add_subdirectory(dim2/Tromp_2005)
add_subdirectory(dim2/poroelastic-semi-infinite)

# 3D
add_subdirectory(dim3/homogeneous_halfspace)
