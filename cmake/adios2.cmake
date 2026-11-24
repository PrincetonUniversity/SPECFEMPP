if (SPECFEM_ENABLE_ADIOS2)

  message(STATUS "ADIOS2 support is enabled. Proceeding with ADIOS2 configuration.")

  # Prepend the CMAKE_MESSAGE_INDENT variable to ensure proper indentation in messages
  list(APPEND CMAKE_MESSAGE_INDENT "  ADIOS2: ")

  # First, try to find ADIOS2 system-wide (unless force install is enabled)
  if(NOT SPECFEM_ENABLE_ADIOS2_FORCE_INSTALL)
    find_package(ADIOS2 QUIET)
  endif()

  # If not found or force install is enabled, fetch and build ADIOS2 from source
  if((NOT ADIOS2_FOUND) OR (SPECFEM_ENABLE_ADIOS2_FORCE_INSTALL))

    # Disable unity build for here
    set(SAVE_UNITY_BUILD ${CMAKE_UNITY_BUILD})
    set(CMAKE_UNITY_BUILD OFF)

    message(STATUS "Installing ADIOS2 from GitHub release...")

    set(ADIOS2_VERSION 2.12.1)
    message(STATUS "Downloading and extracting ADIOS2 (${ADIOS2_VERSION}) library sources. This will take <1 min.")
    include(FetchContent)

    # Set common FetchContent parameters
    set(ADIOS2_URL "https://github.com/ornladios/ADIOS2/archive/refs/tags/v${ADIOS2_VERSION}.tar.gz")

    # Setting the ADIOS2 options
    set(ADIOS2_USE_Fortran ON CACHE BOOL "Enable Fortran support" FORCE)
    set(ADIOS2_USE_Python OFF CACHE BOOL "Disable Python support" FORCE)
    set(ADIOS2_USE_HDF5 OFF CACHE BOOL "Disable HDF5 support" FORCE)
    set(ADIOS2_USE_MPI ${SPECFEM_ENABLE_MPI} CACHE BOOL "Enable MPI support" FORCE)
    set(BUILD_TESTING OFF CACHE BOOL "Disable ADIOS2 testing" FORCE)
    set(ADIOS2_BUILD_EXAMPLES OFF CACHE BOOL "Disable ADIOS2 examples" FORCE)

    # Use external yaml-cpp (our FetchContent version) to avoid conflicts
    set(ADIOS2_USE_EXTERNAL_YAMLCPP ON CACHE BOOL "Use external yaml-cpp" FORCE)

    # ADIOS2 2.12 detects an already-defined HDF5_FOUND even when its own
    # HDF5 option is OFF. Keep SPECFEM's HDF5 detection from enabling the
    # ADIOS2 HDF5 engine implicitly.
    if (DEFINED HDF5_FOUND)
        set(_SPECFEM_HDF5_FOUND ${HDF5_FOUND})
        set(_SPECFEM_HDF5_FOUND_DEFINED TRUE)
        set(HDF5_FOUND FALSE)
    endif()

    # Optional: Control other features based on your needs
    set(ADIOS2_USE_ZeroMQ OFF CACHE BOOL "Disable ZeroMQ" FORCE)
    set(ADIOS2_USE_SST ON CACHE BOOL "Enable SST engine" FORCE)

    if (CMAKE_VERSION VERSION_LESS "3.28.0")
        # For CMake versions < 3.28, EXCLUDE_FROM_ALL is not supported in FetchContent_Declare
        # Use the old FetchContent_Populate pattern
        FetchContent_Declare(
            ADIOS2
            URL ${ADIOS2_URL}
            USES_TERMINAL_DOWNLOAD True
            GIT_PROGRESS TRUE
            DOWNLOAD_NO_EXTRACT FALSE
            DOWNLOAD_EXTRACT_TIMESTAMP FALSE
        )

        FetchContent_GetProperties(ADIOS2)
        if(NOT adios2_POPULATED)
            FetchContent_Populate(ADIOS2)
            add_subdirectory(${adios2_SOURCE_DIR} ${adios2_BINARY_DIR} EXCLUDE_FROM_ALL)
        endif()
    else()
        # For CMake versions >= 3.28, EXCLUDE_FROM_ALL is supported in FetchContent_Declare
        # This allows using FetchContent_MakeAvailable with EXCLUDE_FROM_ALL
        
        # For CMake >= 3.30, set policy CMP0169 to NEW to prefer FetchContent_MakeAvailable
        # over the deprecated FetchContent_Populate pattern. Since we use FetchContent_MakeAvailable
        # in this branch, setting the policy explicitly ensures no deprecation warnings.
        if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.30.0")
          cmake_policy(SET CMP0169 NEW)
        endif()
        
        FetchContent_Declare(
            ADIOS2
            URL ${ADIOS2_URL}
            USES_TERMINAL_DOWNLOAD True
            GIT_PROGRESS TRUE
            DOWNLOAD_NO_EXTRACT FALSE
            DOWNLOAD_EXTRACT_TIMESTAMP FALSE
            EXCLUDE_FROM_ALL
        )

        FetchContent_MakeAvailable(ADIOS2)
    endif()

    if (_SPECFEM_HDF5_FOUND_DEFINED)
        set(HDF5_FOUND ${_SPECFEM_HDF5_FOUND})
        unset(_SPECFEM_HDF5_FOUND)
        unset(_SPECFEM_HDF5_FOUND_DEFINED)
    endif()

    message(STATUS "ADIOS2 downloaded and configured.")

    # Create alias target for modern CMake usage
    # When built from source, adios2::adios2 target is available
    if(NOT TARGET adios2)
        if(TARGET adios2::adios2)
            add_library(adios2 ALIAS adios2::adios2)
        endif()
    endif()

    unset(BUILD_TESTING)

    # Restore the original unity build setting
    set(CMAKE_UNITY_BUILD ${SAVE_UNITY_BUILD})

  elseif(ADIOS2_FOUND)
    message(STATUS "Using system-wide ADIOS2")

    # Create alias target for modern CMake usage with system ADIOS2
    # System ADIOS2 typically provides adios2::adios2 target
    if(NOT TARGET adios2)
        if(TARGET adios2::adios2)
            add_library(adios2 ALIAS adios2::adios2)
        else()
            message(WARNING "ADIOS2 found but adios2::adios2 target not available. This may cause linking issues.")
        endif()
    endif()
  else()
    message(STATUS "ADIOS2 not found.")
    set(SPECFEM_ENABLE_ADIOS2 OFF CACHE BOOL "Disable ADIOS2 support" FORCE)
  endif()

  # Pop the indentation for ADIOS2 messages
  list(POP_BACK CMAKE_MESSAGE_INDENT)

else()
  message(STATUS "ADIOS2 support is disabled. Set SPECFEM_ENABLE_ADIOS2 to ON to enable it.")
  set(SPECFEM_ENABLE_ADIOS2 OFF CACHE BOOL "Disable ADIOS2 support" FORCE)
endif()
