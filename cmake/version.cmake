message(STATUS "Configuring Version & Git Info...")

# Prepend the CMAKE_MESSAGE_INDENT variable to ensure proper indentation in messages
list(APPEND CMAKE_MESSAGE_INDENT "  Version Info: ")


find_package(Git QUIET)

# Semver fallback — overridden from git tag by extract_git_version.cmake
set(SPECFEMPP_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(SPECFEMPP_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(SPECFEMPP_VERSION_PATCH ${PROJECT_VERSION_PATCH})

include(${CMAKE_SOURCE_DIR}/cmake/extract_git_version.cmake)

# Warn if the Git tag version differs from the CMakeLists.txt project version
set(_cmake_version "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}")
set(_git_version   "${SPECFEMPP_VERSION_MAJOR}.${SPECFEMPP_VERSION_MINOR}.${SPECFEMPP_VERSION_PATCH}")
if(NOT _cmake_version VERSION_EQUAL _git_version AND NOT SPECFEMPP_GIT_HASH STREQUAL "unknown")
  message(WARNING
    "Version mismatch: project() declares VERSION ${_cmake_version} "
    "but the latest Git tag is v${_git_version}. "
    "The Git tag version will be used in the build.")
endif()

message(STATUS "Version:            ${_git_version}")
message(STATUS "Git hash:           ${SPECFEMPP_GIT_HASH}")
message(STATUS "Git describe:       ${SPECFEMPP_GIT_DESCRIBE}")
message(STATUS "Commits since tag:  ${SPECFEMPP_COMMITS_SINCE_TAG}")
message(STATUS "From release tag:   ${SPECFEMPP_IS_RELEASE}")
message(STATUS "Is dirty:           ${SPECFEMPP_IS_DIRTY}")

# Render the build-time version script (paths and semver baked in at configure
# time; git hash / dirty / commits-since-tag are re-queried on every build).
configure_file(
  ${CMAKE_SOURCE_DIR}/cmake/version_header.cmake.in
  ${CMAKE_BINARY_DIR}/cmake/version_header.cmake
  @ONLY
)


# Pop the indentation for Kokkos messages
list(POP_BACK CMAKE_MESSAGE_INDENT)
