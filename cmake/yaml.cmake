message(STATUS "Configuring YAML library...")

# Prepend the CMAKE_MESSAGE_INDENT variable to ensure proper indentation in messages
list(APPEND CMAKE_MESSAGE_INDENT "  YAML: ")

# Include FetchContent module
include(FetchContent)

# YAML version
set(YAML_CPP_VERSION "0.7.0" CACHE STRING "yaml-cpp version")

set(YAML_URL https://github.com/jbeder/yaml-cpp/archive/refs/tags/yaml-cpp-${YAML_CPP_VERSION}.tar.gz)

# Disable yaml-cpp installation and config generation
set(YAML_CPP_INSTALL OFF CACHE BOOL "Disable yaml-cpp install targets" FORCE)
set(YAML_CPP_BUILD_INSTALL OFF CACHE BOOL "Disable yaml-cpp installation" FORCE)
set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "Disable yaml-cpp tools" FORCE)
set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "Disable yaml-cpp tests" FORCE)
# Fix to make paths set by yaml-cpp-config.cmake discoverable by ADIOS2
set(YAML_CPP_INSTALL_CMAKEDIR "./" CACHE STRING "CMake config install directory for yaml-cpp" FORCE)

if (CMAKE_VERSION VERSION_LESS "3.28.0")
  # For CMake versions < 3.28, EXCLUDE_FROM_ALL is not supported in FetchContent_Declare
  FetchContent_Declare(
    yaml-cpp
    URL ${YAML_URL}
    USES_TERMINAL_DOWNLOAD True
    GIT_PROGRESS TRUE
    DOWNLOAD_NO_EXTRACT FALSE
    DOWNLOAD_EXTRACT_TIMESTAMP FALSE
  )

  FetchContent_GetProperties(yaml-cpp)
  if(NOT yaml-cpp_POPULATED)
    FetchContent_Populate(yaml-cpp)
    add_subdirectory(${yaml-cpp_SOURCE_DIR} ${yaml-cpp_BINARY_DIR} EXCLUDE_FROM_ALL)
  endif()
else()
  # For CMake versions >= 3.28, EXCLUDE_FROM_ALL is supported in FetchContent_Declare
  FetchContent_Declare(
    yaml-cpp
    URL ${YAML_URL}
    USES_TERMINAL_DOWNLOAD True
    GIT_PROGRESS TRUE
    DOWNLOAD_NO_EXTRACT FALSE
    DOWNLOAD_EXTRACT_TIMESTAMP FALSE
    EXCLUDE_FROM_ALL
  )

  FetchContent_MakeAvailable(yaml-cpp)
endif()

message(STATUS "yaml-cpp library has been configured and is available.")
message(STATUS "Include directory: ${yaml-cpp_SOURCE_DIR}/include")
message(STATUS "Binary directory: ${yaml-cpp_BINARY_DIR}")
# Ensure that the yaml-cpp include directories are added before others to avoid conflicts
# they must match the fetchcontent names above
# yaml-cpp -> yaml-cpp_BINARY_DIR, yaml-cpp_SOURCE_DIR
include_directories(BEFORE SYSTEM ${yaml-cpp_BINARY_DIR} ${yaml-cpp_SOURCE_DIR}/include)

# Overwrite the yaml-cpp config file to ensure it is correctly set up to be used by ADIOS2
file(WRITE ${yaml-cpp_BINARY_DIR}/yaml-cpp-config.cmake "
# Generated config for yaml-cpp from FetchContent
add_library(yaml-cpp::yaml-cpp ALIAS yaml-cpp)
set(yaml-cpp_FOUND TRUE)
set(yaml-cpp_VERSION \"0.7.0\")
")

set(yaml-cpp_DIR ${yaml-cpp_BINARY_DIR} CACHE PATH "Path to yaml-cpp config file directory")

message(STATUS "yaml-cpp include directory set to: ${YAML_CPP_INCLUDE_DIR}")

# pop the indentation for YAML messages
list(POP_BACK CMAKE_MESSAGE_INDENT)
