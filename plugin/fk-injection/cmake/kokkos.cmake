# Private, static, host-only Kokkos for the FK plugin.
#
# Linked into the plugin module with hidden symbols, so it never clashes with
# the Kokkos inside the SPECFEM++ executable. Always fetched: an external
# Kokkos (Kokkos_ROOT) may be shared or device-enabled.

foreach(_backend CUDA HIP SYCL OPENMPTARGET)
  if(Kokkos_ENABLE_${_backend})
    message(FATAL_ERROR
      "FK plugin: Kokkos_ENABLE_${_backend} is not supported. The plugin's "
      "private Kokkos is host-only; two device runtimes in one process are "
      "not supported.")
  endif()
  set(Kokkos_ENABLE_${_backend} OFF CACHE BOOL "" FORCE)
endforeach()

set(Kokkos_ENABLE_SERIAL ON CACHE BOOL "" FORCE)
set(Kokkos_ENABLE_OPENMP ${FK_PLUGIN_ENABLE_OPENMP} CACHE BOOL "" FORCE)
set(KOKKOS_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF)

# Same workaround as core's cmake/kokkos.cmake: with LIBDL on, Kokkos adds the
# Xcode SDK include dir, which breaks Homebrew LLVM's <cmath>.
if(APPLE)
  set(Kokkos_ENABLE_LIBDL OFF CACHE BOOL "" FORCE)
endif()

# Keep in sync with KOKKOS_VERSION in core's cmake/kokkos.cmake.
set(FK_PLUGIN_KOKKOS_VERSION "5.2.0")

include(FetchContent)
FetchContent_Declare(kokkos
  URL https://github.com/kokkos/kokkos/archive/${FK_PLUGIN_KOKKOS_VERSION}.zip
  DOWNLOAD_EXTRACT_TIMESTAMP FALSE
  EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(kokkos)
