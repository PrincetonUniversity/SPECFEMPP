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

add_library(fk_injection_impl STATIC
  src/fk_injection_plugin.cpp
  src/field_sampler.cpp
  src/impl/depth_series.cpp
  src/impl/layer_stack.cpp
  src/layered_model.cpp
  src/model_file_reader.cpp
  src/plane_wave_response.cpp
  src/plugin_config.cpp
  src/series_table.cpp
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
# SPECFEM++'s yaml-cpp is linked statically; the SDK's export control keeps
# its symbols from clashing with the copy inside the executable.
target_link_libraries(fk_injection_impl PUBLIC specfem::plugin_sdk Kokkos::kokkos
                                        PRIVATE yaml-cpp::yaml-cpp)
# Transforms run serially; parallelism, if any, is over sampled points.
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
