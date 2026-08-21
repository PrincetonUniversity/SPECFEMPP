cmake_minimum_required(VERSION 3.10)

project(GLOBE_EVALUATOR LANGUAGES Fortran C)

# The vendored globe sources are Fortran 95; these wrappers use ISO_C_BINDING,
# which is Fortran 2003. Set the standard explicitly rather than inheriting the
# 95 setting used inside fortran/meshfem3d_globe/.
set(CMAKE_Fortran_STANDARD 2003)
set(CMAKE_Fortran_STANDARD_REQUIRED ON)

if(NOT TARGET MPI::MPI_Fortran)
  find_package(MPI REQUIRED COMPONENTS Fortran)
endif()

set(CMAKE_Fortran_MODULE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/modules)

add_library(globe_evaluator globe_model_evaluator.F90)
add_library(globe_evaluator::globe_evaluator ALIAS globe_evaluator)

# The C ABI header, for consumers.
target_include_directories(globe_evaluator PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Fortran .mod files are NOT propagated through CMake targets, and the globe
# tree publishes them only via a directory-scoped include_directories() at
# fortran/meshfem3d_globe/CMakeLists.txt:40 -- which this sibling directory does
# not inherit. So point at the globe's module output directory explicitly. The
# path must track CMAKE_Fortran_MODULE_DIRECTORY as set in
# fortran/meshfem3d_globe/CMakeLists.txt:35.
target_include_directories(globe_evaluator PRIVATE
  ${CMAKE_BINARY_DIR}/fortran/meshfem3d_globe/modules
)

# PUBLIC, not PRIVATE: a C++ consumer needs the whole Fortran archive chain and
# the Fortran runtime transitively, not just this object.
target_link_libraries(globe_evaluator PUBLIC
  meshfem3D_globe::mesh
  meshfem3D_globe::mesh_module
  meshfem3D_globe::shared
  meshfem3D_globe::shared_module
  meshfem3D_globe::constants
  # model_gll.f90 and model_attenuation_gll.f90 reference read_gll_model_adios /
  # read_gll_qmu_model_adios unconditionally, but the *_adios.F90 sources that
  # define them are not in MESHFEM3D_GLOBE_MESH_SOURCES. xmeshfem3D_globe
  # resolves them via these no-op stubs; any consumer of
  # meshfem3D_globe::mesh needs them too.
  meshfem3D_globe::adios_methods_stubs
  meshfem3D_globe::asdf_methods_stubs
  MPI::MPI_Fortran
)

if(SPECFEM_INSTALL)
  install(TARGETS globe_evaluator
    ARCHIVE DESTINATION lib
    COMPONENT libraries
  )
endif()
