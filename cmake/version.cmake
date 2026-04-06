find_package(Git QUIET)

# Defaults — used as fallback when Git is unavailable or repo has no tags
set(SPECFEMPP_VERSION_MAJOR     ${PROJECT_VERSION_MAJOR})
set(SPECFEMPP_VERSION_MINOR     ${PROJECT_VERSION_MINOR})
set(SPECFEMPP_VERSION_PATCH     ${PROJECT_VERSION_PATCH})
set(SPECFEMPP_GIT_HASH          "unknown")
set(SPECFEMPP_GIT_DESCRIBE      "unknown")
set(SPECFEMPP_COMMITS_SINCE_TAG 0)
set(SPECFEMPP_IS_RELEASE        0)
set(SPECFEMPP_IS_DIRTY          0)
set(SPECFEMPP_HAS_GIT_INFO      0)

if(GIT_FOUND)
  execute_process(
    COMMAND ${GIT_EXECUTABLE} describe --tags --match "v*" --long --dirty
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE _git_describe
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
    RESULT_VARIABLE _git_describe_result
  )
  execute_process(
    COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE SPECFEMPP_GIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )

  if(_git_describe_result EQUAL 0 AND _git_describe)
    set(SPECFEMPP_HAS_GIT_INFO 1)
    set(SPECFEMPP_GIT_DESCRIBE "${_git_describe}")

    # git describe --long output: v<maj>.<min>.<patch>-<N>-g<hash>[-dirty]
    if(_git_describe MATCHES
        "^v([0-9]+)\\.([0-9]+)\\.([0-9]+)-([0-9]+)-g([0-9a-f]+)-dirty$")
      set(SPECFEMPP_VERSION_MAJOR     "${CMAKE_MATCH_1}")
      set(SPECFEMPP_VERSION_MINOR     "${CMAKE_MATCH_2}")
      set(SPECFEMPP_VERSION_PATCH     "${CMAKE_MATCH_3}")
      set(SPECFEMPP_COMMITS_SINCE_TAG "${CMAKE_MATCH_4}")
      set(SPECFEMPP_GIT_HASH          "${CMAKE_MATCH_5}")
      set(SPECFEMPP_IS_DIRTY          1)
      set(SPECFEMPP_IS_RELEASE        0)

    elseif(_git_describe MATCHES
        "^v([0-9]+)\\.([0-9]+)\\.([0-9]+)-([0-9]+)-g([0-9a-f]+)$")
      set(SPECFEMPP_VERSION_MAJOR     "${CMAKE_MATCH_1}")
      set(SPECFEMPP_VERSION_MINOR     "${CMAKE_MATCH_2}")
      set(SPECFEMPP_VERSION_PATCH     "${CMAKE_MATCH_3}")
      set(SPECFEMPP_COMMITS_SINCE_TAG "${CMAKE_MATCH_4}")
      set(SPECFEMPP_GIT_HASH          "${CMAKE_MATCH_5}")
      set(SPECFEMPP_IS_DIRTY          0)
      if(CMAKE_MATCH_4 EQUAL 0)
        set(SPECFEMPP_IS_RELEASE 1)
      else()
        set(SPECFEMPP_IS_RELEASE 0)
      endif()
    endif()
  endif()
endif()

if(NOT SPECFEMPP_GIT_HASH)
  set(SPECFEMPP_GIT_HASH "unknown")
endif()

# Warn if the Git tag version differs from the CMakeLists.txt project version
set(_cmake_version "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}")
set(_git_version   "${SPECFEMPP_VERSION_MAJOR}.${SPECFEMPP_VERSION_MINOR}.${SPECFEMPP_VERSION_PATCH}")
if(NOT _cmake_version VERSION_EQUAL _git_version AND NOT SPECFEMPP_GIT_HASH STREQUAL "unknown")
  message(WARNING
    "Version mismatch: project() declares VERSION ${_cmake_version} "
    "but the latest Git tag is v${_git_version}. "
    "The Git tag version will be used in the build.")
endif()

message(STATUS "SPECFEM++ version:            ${_git_version}")
message(STATUS "SPECFEM++ git hash:           ${SPECFEMPP_GIT_HASH}")
message(STATUS "SPECFEM++ git describe:       ${SPECFEMPP_GIT_DESCRIBE}")
message(STATUS "SPECFEM++ commits since tag:  ${SPECFEMPP_COMMITS_SINCE_TAG}")
message(STATUS "SPECFEM++ is release:         ${SPECFEMPP_IS_RELEASE}")
message(STATUS "SPECFEM++ is dirty:           ${SPECFEMPP_IS_DIRTY}")
