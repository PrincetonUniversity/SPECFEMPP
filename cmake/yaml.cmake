message(STATUS "Configuring YAML library...")

# Prepend the CMAKE_MESSAGE_INDENT variable to ensure proper indentation in messages
list(APPEND CMAKE_MESSAGE_INDENT "  YAML: ")

# Install yaml content
FetchContent_Declare(
        yaml-cpp
        GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
        GIT_TAG 2f86d13775d119edbb69af52e5f566fd65c6953b
)

# Disable yaml-cpp installation
set(YAML_CPP_BUILD_INSTALL OFF CACHE BOOL "Disable yaml-cpp installation" FORCE)

FetchContent_MakeAvailable(yaml-cpp)

# Ensure that the yaml-cpp include directories are added before others to avoid conflicts
# they must match the fetchcontent names above
# yaml-cpp -> yaml-cpp_BINARY_DIR, yaml-cpp_SOURCE_DIR
include_directories(BEFORE SYSTEM ${yaml-cpp_BINARY_DIR} ${yaml-cpp_SOURCE_DIR}/include)

# pop the indentation for YAML messages
list(POP_BACK CMAKE_MESSAGE_INDENT)
