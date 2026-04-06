# Build Scotch 5.1.12b as part of the SPECFEMPP project
# Scotch is a graph partitioning library used by meshfem2d and meshfem3d

if(WITH_SCOTCH)
  message(STATUS "Configuring Scotch 5.1.12b...")
  list(APPEND CMAKE_MESSAGE_INDENT "  Scotch: ")

  # Temporarily disable unity build for Scotch (plain C, many small files)
  set(_saved_unity_build ${CMAKE_UNITY_BUILD})
  set(CMAKE_UNITY_BUILD OFF)

  add_subdirectory(
    ${CMAKE_SOURCE_DIR}/fortran/scotch_5.1.12b
    ${CMAKE_BINARY_DIR}/scotch
  )

  set(CMAKE_UNITY_BUILD ${_saved_unity_build})

  list(POP_BACK CMAKE_MESSAGE_INDENT)
  message(STATUS "Scotch 5.1.12b configured")
endif()
