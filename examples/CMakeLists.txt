cmake_minimum_required(VERSION 3.17.5)

# 2D
add_subdirectory(anisotropic-crystal)
add_subdirectory(homogeneous-medium-flat-topography)
add_subdirectory(homogeneous-spin)
add_subdirectory(fluid-solid-interface)
add_subdirectory(fluid-solid-bathymetry)
add_subdirectory(solid-solid-interface)
add_subdirectory(Tromp_2005)
add_subdirectory(poroelastic-semi-infinite)
add_subdirectory(spin-in-homogeneous-isotropic-medium)

# 3D
add_subdirectory(dim3/homogeneous_halfspace)
