# FK injection plugin for SPECFEM++.
#
# Built with SPECFEM++ against the in-tree plugin SDK and SPECFEM++'s Kokkos;
# it never links SPECFEM++ core targets or includes core headers. Kokkos is
# used for Kokkos::mdspan only. Tests are added from tests/unit-tests.
#
# Layout mirrors core: everything under core/ goes into specfem_fk_core
# (unit-testable without dlopen); only plugin.cpp is module-only. Host-only.

# Static code is linked into a MODULE.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(FK_PLUGIN_DOUBLE "Use double precision for FK computations" OFF)
option(FK_PLUGIN_ENABLE_OPENMP "Parallelize field sampling with OpenMP" OFF)

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
add_library(specfem_fk_core INTERFACE)
target_include_directories(specfem_fk_core INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/core
)
target_include_directories(specfem_fk_core SYSTEM INTERFACE
  ${pocketfft_SOURCE_DIR}
)
target_link_libraries(specfem_fk_core INTERFACE Kokkos::kokkos)
# Transforms run serially; parallelism, if any, is over depth groups.
target_compile_definitions(specfem_fk_core INTERFACE
  POCKETFFT_NO_MULTITHREADING
)
if(FK_PLUGIN_DOUBLE)
  target_compile_definitions(specfem_fk_core INTERFACE SPECFEM_FK_PLUGIN_DOUBLE)
endif()
if(FK_PLUGIN_ENABLE_OPENMP)
  find_package(OpenMP REQUIRED)
  target_link_libraries(specfem_fk_core INTERFACE OpenMP::OpenMP_CXX)
  target_compile_definitions(specfem_fk_core INTERFACE FK_PLUGIN_ENABLE_OPENMP)
endif()

specfem_add_injection_plugin(specfem_fk_plugin
  SOURCES   plugin.cpp
  LIBRARIES specfem_fk_core
)
