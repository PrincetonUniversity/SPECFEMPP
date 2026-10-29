# Analytic plane-wave injection plugin for SPECFEM++.
#
# A standalone project built against the installed plugin SDK only, as an
# external plugin author would:
#
#   cmake --install <specfem build> --component plugin_sdk --prefix <sdk>
#   cmake -S . -B build -DCMAKE_PREFIX_PATH=<sdk>
#   cmake --build build && ctest --test-dir build

cmake_minimum_required(VERSION 3.20)
project(specfem_plane_wave_plugin LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(SpecfemPluginSDK 1 REQUIRED)

# yaml-cpp is linked statically into the plugin; the SDK's export control keeps
# its symbols from clashing with the copy inside SPECFEM++.
include(FetchContent)
set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_CONTRIB OFF CACHE BOOL "" FORCE)
set(YAML_CPP_INSTALL OFF CACHE BOOL "" FORCE)
set(YAML_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
# yaml-cpp 0.8.0 declares a CMake minimum that CMake 4 rejects without this.
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
FetchContent_Declare(yaml-cpp
  URL https://github.com/jbeder/yaml-cpp/archive/refs/tags/0.8.0.tar.gz
  DOWNLOAD_EXTRACT_TIMESTAMP FALSE
)
FetchContent_MakeAvailable(yaml-cpp)
set_target_properties(yaml-cpp PROPERTIES POSITION_INDEPENDENT_CODE ON)

find_package(OpenMP COMPONENTS CXX)

add_library(plane_wave_impl STATIC
  src/config.cpp
  src/plane_wave_plugin.cpp
)
target_include_directories(plane_wave_impl PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(plane_wave_impl PUBLIC specfem::plugin_sdk
                                      PRIVATE yaml-cpp::yaml-cpp)
set_target_properties(plane_wave_impl PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
)
if(OpenMP_CXX_FOUND)
  target_link_libraries(plane_wave_impl PUBLIC OpenMP::OpenMP_CXX)
endif()

specfem_add_injection_plugin(plane_wave
  SOURCES   src/plugin_entry.cpp
  LIBRARIES plane_wave_impl
)

enable_testing()
add_executable(plane_wave_tests tests/plane_wave_tests.cpp)
target_link_libraries(plane_wave_tests PRIVATE plane_wave_impl)
add_test(NAME plane_wave_tests COMMAND plane_wave_tests)
