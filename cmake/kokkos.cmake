

if (DEFINED Kokkos_ROOT)
    message(STATUS "Using user-defined Kokkos root: ${Kokkos_ROOT}")
    find_package(Kokkos REQUIRED PATHS ${Kokkos_ROOT} NO_DEFAULT_PATH)
    return()
endif()

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

# On macOS, dlopen is part of libSystem and does not require a separate libdl.
# Leaving LIBDL enabled causes Kokkos to locate dlfcn.h inside the Xcode SDK
# and add that SDK's usr/include as an explicit -I path, which puts the SDK
# math.h before Homebrew LLVM's libc++ math.h and breaks the <cmath> header.
if(APPLE)
  set(Kokkos_ENABLE_LIBDL OFF CACHE BOOL "" FORCE)
endif()

# Set the policy for CMake versions > 3.30
if (CMAKE_VERSION VERSION_GREATER "3.30.0")
    # For CMake versions > 3.30, we need to use Set the policy)
    if (CMAKE_VERSION VERSION_LESS "3.28.0")
        cmake_policy(SET CMP0169 OLD)
    else()
        cmake_policy(SET CMP0169 NEW)
    endif()
endif()

if (DEFINED KOKKOS_PATH)
    message(STATUS "Using Kokkos from KOKKOS_PATH: ${KOKKOS_PATH}")
    add_subdirectory(${KOKKOS_PATH} ${CMAKE_CURRENT_BINARY_DIR}/kokkos EXCLUDE_FROM_ALL)
    # Pop the indentation for Kokkos messages
else()

    # Pinned to the Kokkos 5.2.0 release candidate branch
    set(KOKKOS_VERSION "release-candidate-5.2.0")

    # Set common FetchContent parameters
    # set(KOKKOS_URL "https://github.com/kokkos/kokkos/archive/refs/tags/${KOKKOS_VERSION}.zip")
    set(KOKKOS_URL "https://github.com/kokkos/kokkos/archive/${KOKKOS_VERSION}.zip")


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
endif()

# Pop the indentation for Kokkos messages
list(POP_BACK CMAKE_MESSAGE_INDENT)
