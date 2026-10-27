# FK injection plugin for SPECFEM++.
#
# A standalone project built against the installed plugin SDK only; it never
# links SPECFEM++ core targets or includes core headers. Build it with:
#   cmake -S plugins/injection/fk -B build/fk-plugin \
#         -DCMAKE_PREFIX_PATH=<plugin_sdk install prefix>
#
# Layout mirrors core: everything under core/ goes into the static library
# specfem_fk_core (unit-testable without dlopen); only plugin.cpp is
# module-only.

cmake_minimum_required(VERSION 3.20)
project(specfem_fk_plugin LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# The static library and its private Kokkos are linked into a MODULE.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(FK_PLUGIN_DOUBLE "Use double precision for FK computations" OFF)
option(FK_PLUGIN_ENABLE_OPENMP "Enable the OpenMP backend of the private Kokkos" OFF)
option(FK_PLUGIN_BUILD_TESTS "Build the FK plugin tests" ON)

find_package(SpecfemPluginSDK 1 REQUIRED)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/kokkos.cmake)

add_library(specfem_fk_core STATIC
  core/specfem/fk_plugin/impl/kokkos_lifetime.cpp
)
target_include_directories(specfem_fk_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/core
)
target_link_libraries(specfem_fk_core PUBLIC Kokkos::kokkos)
if(FK_PLUGIN_DOUBLE)
  target_compile_definitions(specfem_fk_core PUBLIC SPECFEM_FK_PLUGIN_DOUBLE)
endif()
set_target_properties(specfem_fk_core PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
  UNITY_BUILD OFF
)

specfem_add_injection_plugin(specfem_fk_plugin
  SOURCES   plugin.cpp
  LIBRARIES specfem_fk_core
)

if(FK_PLUGIN_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
