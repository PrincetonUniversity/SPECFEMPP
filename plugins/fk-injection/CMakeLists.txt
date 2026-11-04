# FK injection plugin for SPECFEM++.
#
# Built with SPECFEM++ against the in-tree plugin SDK and SPECFEM++'s Kokkos;
# it never links SPECFEM++ core targets or includes core headers. Kokkos is
# used for Kokkos::mdspan only. Tests are added from tests/unit-tests.
#
# Everything but src/plugin_entry.cpp goes into fk_injection_impl, so the tests
# can link it without the entry point. Host-only.

# Static code is linked into a MODULE.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(FK_INJECTION_DOUBLE "Use double precision for FK computations" OFF)
option(FK_INJECTION_ENABLE_OPENMP "Parallelize field sampling with OpenMP" OFF)

# Header-only FFT (BSD-3), pinned to a commit on its cpp branch.
include(FetchContent)
FetchContent_Declare(pocketfft
  URL https://github.com/mreineck/pocketfft/archive/c90e55b3d529f8efa40ed01a20de22405f45fc65.tar.gz
  URL_HASH SHA256=2d90016da31a8cc1dd0ac0c89c1f2fb55df8f4ccc3f0386dcfbad09ac6d1c3ba
  DOWNLOAD_EXTRACT_TIMESTAMP FALSE
)
FetchContent_MakeAvailable(pocketfft)

# yaml-cpp is linked statically into the plugin; the SDK's export control keeps
# its symbols from clashing with the copy inside SPECFEM++.
set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_CONTRIB OFF CACHE BOOL "" FORCE)
set(YAML_CPP_INSTALL OFF CACHE BOOL "" FORCE)
set(YAML_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
# yaml-cpp 0.8.0 declares a CMake minimum that CMake 4 rejects without this.
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
FetchContent_Declare(yaml-cpp
  URL https://github.com/jbeder/yaml-cpp/archive/refs/tags/0.8.0.tar.gz
  URL_HASH SHA256=fbe74bbdcee21d656715688706da3c8becfd946d92cd44705cc6098bb23b3a16
  DOWNLOAD_EXTRACT_TIMESTAMP FALSE
)
FetchContent_MakeAvailable(yaml-cpp)

add_library(fk_injection_impl STATIC
  src/fk_result.cpp
  src/impl/depth_groups.cpp
  src/impl/depth_series.cpp
  src/impl/driver.cpp
  src/impl/fk_setup.cpp
  src/impl/layer_stack.cpp
  src/layered_model.cpp
  src/model_file_reader.cpp
  src/plugin_config.cpp
  src/solver.cpp
  src/time_window.cpp
)
set_target_properties(fk_injection_impl PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
  UNITY_BUILD OFF
)
target_include_directories(fk_injection_impl PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_include_directories(fk_injection_impl SYSTEM PUBLIC
  ${pocketfft_SOURCE_DIR}
)
# The types speak the SDK's medium enum; Kokkos provides Kokkos::mdspan.
target_link_libraries(fk_injection_impl PUBLIC specfem::plugin_sdk Kokkos::kokkos
                                        PRIVATE yaml-cpp::yaml-cpp)
# Transforms run serially; parallelism, if any, is over depth groups.
target_compile_definitions(fk_injection_impl PUBLIC
  POCKETFFT_NO_MULTITHREADING
)
if(FK_INJECTION_DOUBLE)
  target_compile_definitions(fk_injection_impl PUBLIC FK_INJECTION_DOUBLE_PRECISION)
endif()
if(FK_INJECTION_ENABLE_OPENMP)
  find_package(OpenMP REQUIRED)
  target_link_libraries(fk_injection_impl PUBLIC OpenMP::OpenMP_CXX)
  target_compile_definitions(fk_injection_impl PUBLIC FK_INJECTION_ENABLE_OPENMP)
endif()

specfem_add_injection_plugin(fk_injection
  SOURCES   src/plugin_entry.cpp
  LIBRARIES fk_injection_impl
)
