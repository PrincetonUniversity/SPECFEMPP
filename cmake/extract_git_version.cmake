# Shared git version extraction logic.
#
# Caller must set BEFORE including this file:
#   GIT_EXECUTABLE      — path to git (empty/NOTFOUND disables git queries)
#   CMAKE_SOURCE_DIR    — repo root
#   SPECFEM_VERSION_MAJOR/MINOR/PATCH  — semver fallback values
#
# This file sets / overwrites:
#   SPECFEM_GIT_HASH          short hash of HEAD  ("unknown" on failure)
#   SPECFEM_GIT_DESCRIBE      raw git-describe output
#   SPECFEM_COMMITS_SINCE_TAG number of commits ahead of nearest tag
#   SPECFEM_IS_RELEASE        1 if HEAD is exactly on a clean tag
#   SPECFEM_IS_DIRTY          1 if working tree has uncommitted changes
#   SPECFEM_HAS_GIT_INFO      1 if git describe succeeded
#   SPECFEM_VERSION_MAJOR/MINOR/PATCH  overridden from tag when available

set(SPECFEM_GIT_HASH          "unknown")
set(SPECFEM_GIT_DESCRIBE      "unknown")
set(SPECFEM_COMMITS_SINCE_TAG 0)
set(SPECFEM_IS_RELEASE        0)
set(SPECFEM_IS_DIRTY          0)
set(SPECFEM_HAS_GIT_INFO      0)

if(GIT_EXECUTABLE)
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" describe --tags --match "v*" --long --dirty
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    OUTPUT_VARIABLE _git_describe
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
    RESULT_VARIABLE _git_describe_result
  )
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" rev-parse --short HEAD
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    OUTPUT_VARIABLE SPECFEM_GIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )

  if(_git_describe_result EQUAL 0 AND _git_describe)
    set(SPECFEM_HAS_GIT_INFO 1)
    set(SPECFEM_GIT_DESCRIBE "${_git_describe}")

    # v<maj>.<min>.<patch>-<N>-g<hash>-dirty
    if(_git_describe MATCHES
        "^v([0-9]+)\\.([0-9]+)\\.([0-9]+)-([0-9]+)-g([0-9a-f]+)-dirty$")
      set(SPECFEM_VERSION_MAJOR     "${CMAKE_MATCH_1}")
      set(SPECFEM_VERSION_MINOR     "${CMAKE_MATCH_2}")
      set(SPECFEM_VERSION_PATCH     "${CMAKE_MATCH_3}")
      set(SPECFEM_COMMITS_SINCE_TAG "${CMAKE_MATCH_4}")
      set(SPECFEM_GIT_HASH          "${CMAKE_MATCH_5}")
      set(SPECFEM_IS_DIRTY          1)
      set(SPECFEM_IS_RELEASE        0)

    # v<maj>.<min>.<patch>-<N>-g<hash>
    elseif(_git_describe MATCHES
        "^v([0-9]+)\\.([0-9]+)\\.([0-9]+)-([0-9]+)-g([0-9a-f]+)$")
      set(SPECFEM_VERSION_MAJOR     "${CMAKE_MATCH_1}")
      set(SPECFEM_VERSION_MINOR     "${CMAKE_MATCH_2}")
      set(SPECFEM_VERSION_PATCH     "${CMAKE_MATCH_3}")
      set(SPECFEM_COMMITS_SINCE_TAG "${CMAKE_MATCH_4}")
      set(SPECFEM_GIT_HASH          "${CMAKE_MATCH_5}")
      set(SPECFEM_IS_DIRTY          0)
      if(CMAKE_MATCH_4 EQUAL 0)
        set(SPECFEM_IS_RELEASE 1)
      else()
        set(SPECFEM_IS_RELEASE 0)
      endif()
    endif()
  endif()
endif()

if(NOT SPECFEM_GIT_HASH)
  set(SPECFEM_GIT_HASH "unknown")
endif()
