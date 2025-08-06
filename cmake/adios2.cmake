if (SPECFEM_ENABLE_ADIOS2)

  # Disable unity build for here
  set(SAVE_UNITY_BUILD ${CMAKE_UNITY_BUILD})
  set(CMAKE_UNITY_BUILD OFF)

  message(STATUS "Enabling ADIOS2 support")
  list(APPEND CMAKE_MESSAGE_INDENT "  ADIOS2: ")

  set(ADIOS2_VERSION 2.10.2)
  message(STATUS "Downloading and extracting ADIOS2 (${ADIOS2_VERSION}) library sources. This will take <1 min.")
  include(FetchContent)

    # Fetch ADIOS2 from the Github release zip file to reduce download time
  FetchContent_Declare(
      ADIOS2
      URL https://github.com/ornladios/ADIOS2/archive/refs/tags/v${ADIOS2_VERSION}.tar.gz
      USES_TERMINAL_DOWNLOAD True
      GIT_PROGRESS TRUE
      DOWNLOAD_NO_EXTRACT FALSE
      DOWNLOAD_EXTRACT_TIMESTAMP FALSE
      EXCLUDE_FROM_ALL
  )

  # Setting the ADIOS2 options
  set(ADIOS2_USE_Fortran ON CACHE BOOL "Enable Fortran support" FORCE)
  set(ADIOS2_USE_Python OFF CACHE BOOL "Disable Python support" FORCE)
  set(ADIOS2_USE_HDF5 OFF CACHE BOOL "Disable HDF5 support" FORCE)
  set(ADIOS2_USE_MPI ${SPECFEM_ENABLE_MPI} CACHE BOOL "Enable MPI support" FORCE)
  set(BUILD_TESTING OFF CACHE BOOL "Disable ADIOS2 testing" FORCE)
  set(ADIOS2_BUILD_EXAMPLES OFF CACHE BOOL "Disable ADIOS2 examples" FORCE)
  set(ADIOS2_USE_EXTERNAL_YAMLCPP ON CACHE BOOL "Use external YAML-CPP" FORCE)
  set(ADIOS2_USE_EXTERNAL_PUGIXML ON CACHE BOOL "Use external pugixml" FORCE)

  # Optional: Control other features based on your needs
  set(ADIOS2_USE_ZeroMQ OFF CACHE BOOL "Disable ZeroMQ" FORCE)
  set(ADIOS2_USE_SST ON CACHE BOOL "Enable SST engine" FORCE)

  FetchContent_MakeAvailable(ADIOS2)

  message(STATUS "ADIOS2 downloaded and configured.")

  unset(BUILD_TESTING)

  list(POP_BACK CMAKE_MESSAGE_INDENT)

  set(CMAKE_UNITY_BUILD ${SAVE_UNITY_BUILD})

else()
  message(STATUS "ADIOS2 support is disabled. Set SPECFEM_ENABLE_ADIOS2 to ON to enable it.")
  set(SPECFEM_ENABLE_ADIOS2 OFF CACHE BOOL "Disable ADIOS2 support" FORCE)
endif()
