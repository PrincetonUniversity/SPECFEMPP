# Header-only SDK for SPECFEM++ wavefield-injection plugins.
#
# Deliberately independent of every core target (Kokkos, MPI, yaml-cpp): plugin
# authors build against the installed SDK alone.
#
# Install it with:
#   cmake --install <build> --component plugin_sdk --prefix <dir>
# It is excluded from a plain `cmake --install`, so existing installs are unchanged.

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

add_library(specfem_plugin_sdk INTERFACE)
add_library(specfem::plugin_sdk ALIAS specfem_plugin_sdk)
set_target_properties(specfem_plugin_sdk PROPERTIES EXPORT_NAME plugin_sdk)
target_include_directories(specfem_plugin_sdk INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_compile_features(specfem_plugin_sdk INTERFACE cxx_std_20)

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/SpecfemPluginSDKFunctions.cmake")

# The package version is the ABI version, read from the header so there is a
# single source of truth.
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/include/specfem/plugin/injection_abi.h"
     _abi_line REGEX "^#define SPECFEM_INJECTION_ABI_VERSION [0-9]+")
string(REGEX REPLACE ".* ([0-9]+)$" "\\1" SPECFEM_PLUGIN_ABI_VERSION "${_abi_line}")

set(_config_dir "${CMAKE_INSTALL_LIBDIR}/cmake/SpecfemPluginSDK")

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/SpecfemPluginSDKConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/SpecfemPluginSDKConfig.cmake"
    INSTALL_DESTINATION "${_config_dir}"
)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/SpecfemPluginSDKConfigVersion.cmake"
    VERSION ${SPECFEM_PLUGIN_ABI_VERSION}
    COMPATIBILITY SameMajorVersion
    ARCH_INDEPENDENT
)

install(TARGETS specfem_plugin_sdk
    EXPORT SpecfemPluginSDKTargets
    COMPONENT plugin_sdk
    EXCLUDE_FROM_ALL
)
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    COMPONENT plugin_sdk
    EXCLUDE_FROM_ALL
)
install(EXPORT SpecfemPluginSDKTargets
    NAMESPACE specfem::
    DESTINATION "${_config_dir}"
    COMPONENT plugin_sdk
    EXCLUDE_FROM_ALL
)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/SpecfemPluginSDKConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/SpecfemPluginSDKConfigVersion.cmake"
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/SpecfemPluginSDKFunctions.cmake"
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/check_plugin_exports.cmake"
    DESTINATION "${_config_dir}"
    COMPONENT plugin_sdk
    EXCLUDE_FROM_ALL
)
