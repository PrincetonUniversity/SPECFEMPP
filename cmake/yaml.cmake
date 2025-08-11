message(STATUS "Configuring YAML library...")

# Prepend the CMAKE_MESSAGE_INDENT variable to ensure proper indentation in messages
list(APPEND CMAKE_MESSAGE_INDENT "  YAML: ")

# Install yaml content
FetchContent_Declare(
        yaml-cpp
        GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
        GIT_TAG 2f86d13775d119edbb69af52e5f566fd65c6953b
)

# Disable yaml-cpp installation and config generation
set(YAML_CPP_BUILD_INSTALL OFF CACHE BOOL "Disable yaml-cpp installation" FORCE)
set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "Disable yaml-cpp tools" FORCE)
set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "Disable yaml-cpp tests" FORCE)
set(YAML_CPP_INSTALL OFF CACHE BOOL "Disable yaml-cpp install targets" FORCE)
# Fix to make paths set by yaml-cpp-config.cmake discoverable by ADIOS2
set(YAML_CPP_INSTALL_CMAKEDIR "./" CACHE STRING "CMake config install directory for yaml-cpp" FORCE)

FetchContent_MakeAvailable(yaml-cpp)

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
set(yaml-cpp_VERSION \"0.8.0\")
")

message(STATUS "yaml-cpp include directory set to: ${YAML_CPP_INCLUDE_DIR}")

# pop the indentation for YAML messages
list(POP_BACK CMAKE_MESSAGE_INDENT)
