

message(STATUS "Configuring Kokkos library...")

# Prepend the CMAKE_MESSAGE_INDENT variable to ensure proper indentation in messages
list(APPEND CMAKE_MESSAGE_INDENT "  Kokkos: ")

if (DEFINED Kokkos_ENABLE_CUDA)
    if (Kokkos_ENABLE_CUDA)
        # message(STATUS "Setting CUDA variables")
        set(Kokkos_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE ON CACHE BOOL "Using CUDA Relocatable device by default")
        set(Kokkos_ENABLE_CUDA_CONSTEXPR ON CACHE BOOL "Using CUDA Constexpr by default")
    endif()
endif()

# Install Kokkos as a dependency
# Set Kokkos options before fetching
set(KOKKOS_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)

# Set the policy for CMake versions > 3.30
if (CMAKE_VERSION VERSION_GREATER "3.30.0")
    # For CMake versions > 3.30, we need to use Set the policy)
    if (CMAKE_VERSION VERSION_LESS "3.28.0")
        cmake_policy(SET CMP0169 OLD)
    else()
        cmake_policy(SET CMP0169 NEW)
    endif()
endif()

# Set common FetchContent parameters
set(KOKKOS_URL "https://github.com/kokkos/kokkos/archive/refs/tags/4.6.01.zip")

# For CMake versions < 3.28, EXCLUDE_FROM_ALL is not supported in FetchContent_Declare
if (CMAKE_VERSION VERSION_LESS "3.28.0")
    FetchContent_Declare(kokkos DOWNLOAD_EXTRACT_TIMESTAMP FALSE URL ${KOKKOS_URL})

    FetchContent_GetProperties(kokkos)
    if(NOT kokkos_POPULATED)
        FetchContent_Populate(kokkos)
        add_subdirectory(${kokkos_SOURCE_DIR} ${kokkos_BINARY_DIR} EXCLUDE_FROM_ALL)
    endif()

# For CMake versions >= 3.28, EXCLUDE_FROM_ALL is supported in FetchContent_Declare
else()

    FetchContent_Declare(kokkos DOWNLOAD_EXTRACT_TIMESTAMP FALSE URL ${KOKKOS_URL} EXCLUDE_FROM_ALL)
    FetchContent_MakeAvailable(kokkos)
endif()

# Pop the indentation for Kokkos messages
list(POP_BACK CMAKE_MESSAGE_INDENT)
