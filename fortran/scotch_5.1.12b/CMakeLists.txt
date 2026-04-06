cmake_minimum_required(VERSION 3.10)
project(scotch LANGUAGES C)

# Scotch version
set(SCOTCH_VERSION 5)
set(SCOTCH_RELEASE 1)
set(SCOTCH_PATCHLEVEL 12)

# Global compile definitions required by Scotch
add_definitions(
  -Drestrict=__restrict
  -DSCOTCH_RENAME
  -DSCOTCH_RENAME_PARSER
  -DCOMMON_RANDOM_FIXED_SEED
)

# Platform-specific definitions
if(APPLE)
  add_definitions(-DCOMMON_OS_MACOS)
endif()

# Optional dependencies
find_package(ZLIB QUIET)
if(ZLIB_FOUND)
  add_definitions(-DCOMMON_FILE_COMPRESS_GZ)
  message(STATUS "Scotch: zlib found — enabling gzip support")
else()
  message(STATUS "Scotch: zlib not found — gzip support disabled")
endif()

find_package(Threads QUIET)
if(Threads_FOUND)
  add_definitions(-DCOMMON_PTHREAD)
  message(STATUS "Scotch: pthreads found — enabling thread support")
else()
  message(STATUS "Scotch: pthreads not found — thread support disabled")
endif()

find_package(BISON QUIET)
find_package(FLEX QUIET)

# Directory for generated public headers (scotch.h, scotchf.h)
set(GENERATED_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/include)
file(MAKE_DIRECTORY ${GENERATED_INCLUDE_DIR})

# Build the library
add_subdirectory(src/libscotch)
