# Install yaml content
FetchContent_Declare(
        yaml
        GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
        GIT_TAG 2f86d13775d119edbb69af52e5f566fd65c6953b
)

# Disable yaml-cpp installation
set(YAML_CPP_BUILD_INSTALL OFF CACHE BOOL "Disable yaml-cpp installation" FORCE)

FetchContent_MakeAvailable(yaml)

include_directories(BEFORE SYSTEM ${yaml_BINARY_DIR} ${yaml_SOURCE_DIR}/include)
