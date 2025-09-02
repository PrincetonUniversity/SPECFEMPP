if (SPECFEM_ENABLE_ADIOS2)

  # Disable unity build for here
  set(SAVE_UNITY_BUILD ${CMAKE_UNITY_BUILD})
  set(CMAKE_UNITY_BUILD OFF)

  message(STATUS "Enabling ADIOS2 support")
  list(APPEND CMAKE_MESSAGE_INDENT "  ADIOS2: ")


  if (CMAKE_VERSION VERSION_GREATER "3.30.0")
  # For CMake versions > 3.30, we need to use Set the policy)
    if (CMAKE_VERSION VERSION_LESS "3.28.0")
      cmake_policy(SET CMP0169 OLD)
    else()
      cmake_policy(SET CMP0169 NEW)
    endif()
  endif()

  set(ADIOS2_VERSION 2.10.2)
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

  # Optional: Control other features based on your needs
  set(ADIOS2_USE_ZeroMQ OFF CACHE BOOL "Disable ZeroMQ" FORCE)
  set(ADIOS2_USE_SST ON CACHE BOOL "Enable SST engine" FORCE)

  if (CMAKE_VERSION VERSION_LESS "3.28.0")
      # For CMake versions < 3.28, EXCLUDE_FROM_ALL is not supported in FetchContent_Declare
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

  message(STATUS "ADIOS2 downloaded and configured.")

  # Create alias target for modern CMake usage
  if(NOT TARGET adios2)
      add_library(adios2 ALIAS adios2::adios2)
  endif()

  unset(BUILD_TESTING)

  list(POP_BACK CMAKE_MESSAGE_INDENT)

  set(CMAKE_UNITY_BUILD ${SAVE_UNITY_BUILD})

else()
  message(STATUS "ADIOS2 support is disabled. Set SPECFEM_ENABLE_ADIOS2 to ON to enable it.")
  set(SPECFEM_ENABLE_ADIOS2 OFF CACHE BOOL "Disable ADIOS2 support" FORCE)
endif()
