
if (DEFINED Kokkos_ENABLE_CUDA)
    if (Kokkos_ENABLE_CUDA)
        # message(STATUS "Setting CUDA variables")
        set(Kokkos_ENABLE_CUDA_LAMBDA ON CACHE BOOL "Using CUDA Lambda by default")
        set(Kokkos_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE ON CACHE BOOL "Using CUDA Relocatable device by default")
        set(Kokkos_ENABLE_CUDA_CONSTEXPR ON CACHE BOOL "Using CUDA Constexpr by default")
    endif()
endif()

# Install Kokkos as a dependency
# Set Kokkos options before fetching
set(KOKKOS_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
        kokkos
        DOWNLOAD_EXTRACT_TIMESTAMP FALSE
        URL https://github.com/kokkos/kokkos/archive/refs/tags/4.6.01.zip
)

# Using this instead of FetchContent_MakeAvailable to be backwards compatible
# with older CMake versions
FetchContent_GetProperties(kokkos)
if(NOT kokkos_POPULATED)
  FetchContent_Populate(kokkos)
  add_subdirectory(${kokkos_SOURCE_DIR} ${kokkos_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()
