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

# Header-only until the first source file arrives; then STATIC with hidden
# visibility.
add_library(fk_injection_impl INTERFACE)
target_include_directories(fk_injection_impl INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_include_directories(fk_injection_impl SYSTEM INTERFACE
  ${pocketfft_SOURCE_DIR}
)
target_link_libraries(fk_injection_impl INTERFACE Kokkos::kokkos)
# Transforms run serially; parallelism, if any, is over depth groups.
target_compile_definitions(fk_injection_impl INTERFACE
  POCKETFFT_NO_MULTITHREADING
)
if(FK_INJECTION_DOUBLE)
  target_compile_definitions(fk_injection_impl INTERFACE FK_INJECTION_DOUBLE_PRECISION)
endif()
if(FK_INJECTION_ENABLE_OPENMP)
  find_package(OpenMP REQUIRED)
  target_link_libraries(fk_injection_impl INTERFACE OpenMP::OpenMP_CXX)
  target_compile_definitions(fk_injection_impl INTERFACE FK_INJECTION_ENABLE_OPENMP)
endif()

specfem_add_injection_plugin(fk_injection
  SOURCES   src/plugin_entry.cpp
  LIBRARIES fk_injection_impl
)
