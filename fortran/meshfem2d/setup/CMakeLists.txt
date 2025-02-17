project(FortranTest LANGUAGES Fortran)
cmake_minimum_required(VERSION 3.10)

message(STATUS "-- Configuring meshfem2D headers...")

enable_language(Fortran)
enable_language(C)

include(CheckIncludeFile)
include(CheckFunctionExists)
include(CheckSymbolExists)
include(CheckCSourceCompiles)
include(CheckCCompilerFlag)
include(CheckFortranCompilerFlag)

# Check for headers
foreach(header inttypes.h pthread.h scotch.h stdint.h stdio.h stdlib.h strings.h string.h sys/stat.h sys/types.h unistd.h xmmintrin.h emmintrin.h err.h)
    string(TOUPPER "${header}" UPPER_HEADER)
    string(REPLACE "." "_" UPPER_HEADER "${UPPER_HEADER}")
    check_include_file(${header} HAVE_${UPPER_HEADER})
endforeach()


# Define package information
set(PACKAGE_NAME "SPECFEM2D")
set(PACKAGE_VERSION "8.0.0")
set(PACKAGE_STRING "${PACKAGE_NAME} ${PACKAGE_VERSION}")
set(PACKAGE_TARNAME "specfem2d")
set(PACKAGE_URL "https://github.com/geodynamics/specfem2d")
set(PACKAGE_BUGREPORT "support@specfem2d.org")

# Git-related details (can be dynamically retrieved)
set(SPECFEM2D_GIT_BRANCH "main")  # You can use 'git rev-parse --abbrev-ref HEAD' dynamically
set(SPECFEM2D_GIT_DATE "2024-02-11")
set(SPECFEM2D_GIT_HASH "abcd1234")
set(SPECFEM2D_GIT_REVISION "abcd1234")
set(SPECFEM2D_VERSION "${PACKAGE_VERSION}")
set(SPECFEM2D_RELEASE_VERSION 1)

set(GIT_PACKAGE_VERSION "v8.0.0-11-gf8c66778")
set(GIT_COMMIT_VERSION "f8c66778e3bcff99be726113a1aca338255ed87e")
set(GIT_DATE_VERSION "2023-03-21 19:54:51 +0100")


# ====== FIGURE OUT THE FORTRAN NAME MANGLING START ===========

# Find Fortran C Interface
include(FortranCInterface)

# Write the FC_MANGLING.h file to detect Fortran name mangling
# We can write this for debugging the mangling.
# FortranCInterface_HEADER(${CMAKE_CURRENT_BINARY_DIR}/FC_MANGLING.h MACRO_NAMESPACE "FC_")

FortranCInterface_VERIFY()
# For regular symbols (FC_FUNC)
set(FC_FUNC "")

# Set name or NAME based on case modifier
if(FortranCInterface_GLOBAL_CASE STREQUAL "UPPER")
    string(APPEND FC_FUNC "${FortranCInterface_GLOBAL_PREFIX}NAME##${FortranCInterface_GLOBAL_MIDDLE}${FortranCInterface_GLOBAL_SUFFIX}")
else()
    string(APPEND FC_FUNC "${FortranCInterface_GLOBAL_PREFIX}name##${FortranCInterface_GLOBAL_MIDDLE}${FortranCInterface_GLOBAL_SUFFIX}")
endif()

# For symbols with underscores (FC_FUNC_)
set(FC_FUNC_ "")

# Set name or NAME based on case modifier
if(FortranCInterface_GLOBAL__CASE STREQUAL "UPPER")
    string(APPEND FC_FUNC_ "${FortranCInterface_GLOBAL__PREFIX}NAME##${FortranCInterface_GLOBAL__MIDDLE}${FortranCInterface_GLOBAL__SUFFIX}")
else()
    string(APPEND FC_FUNC_ "${FortranCInterface_GLOBAL__PREFIX}name##${FortranCInterface_GLOBAL__MIDDLE}${FortranCInterface_GLOBAL__SUFFIX}")
endif()

message(STATUS "Generated FC_FUNC(name,NAME) ${FC_FUNC}")
message(STATUS "Generated FC_FUNC_(name,NAME) ${FC_FUNC_}")

# ===================== config.h.in START =====================

# Generate config.h file
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/config.h.in
               ${CMAKE_CURRENT_BINARY_DIR}/config.h @ONLY)

# ===================== config.h.in END =====================

# ===================== config.fh.in START ==================
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/config.fh.in
               ${CMAKE_CURRENT_BINARY_DIR}/config.fh @ONLY)
# ===================== config.fh.in END ====================


# ===================== constants.h.in START ================
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/constants.h.in
               ${CMAKE_CURRENT_BINARY_DIR}/constants.h @ONLY)
# ===================== constants.h.in END ==================

# ===================== precision.h.in START ================
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/precision.h.in
               ${CMAKE_CURRENT_BINARY_DIR}/precision.h @ONLY)
# ===================== precision.h.in END ==================


# ===================== version.fh.in START ================

# Configure the config.h file
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/version.fh.in
               ${CMAKE_CURRENT_BINARY_DIR}/version.fh @ONLY)

# ===================== version.h.in START ================

# Set Header interfaces
add_library(meshfem2D_constants INTERFACE ${CMAKE_CURRENT_BINARY_DIR}/constants.h)
add_library(meshfem2D::constants ALIAS meshfem2D_constants)
target_include_directories(meshfem2D_constants INTERFACE ${CMAKE_CURRENT_BINARY_DIR})

add_library(meshfem2D_precision INTERFACE ${CMAKE_CURRENT_BINARY_DIR}/precision.h)
add_library(meshfem2D::precision ALIAS meshfem2D_precision)
target_include_directories(meshfem2D_precision INTERFACE ${CMAKE_CURRENT_BINARY_DIR})

add_library(meshfem2D_version INTERFACE ${CMAKE_CURRENT_BINARY_DIR}/version.fh)
add_library(meshfem2D::version ALIAS meshfem2D_version)
target_include_directories(meshfem2D_version INTERFACE ${CMAKE_CURRENT_BINARY_DIR})

add_library(meshfem2D_config INTERFACE ${CMAKE_CURRENT_BINARY_DIR}/config.h)
add_library(meshfem2D::config ALIAS meshfem2D_config)
target_include_directories(meshfem2D_config INTERFACE ${CMAKE_CURRENT_BINARY_DIR})

add_library(meshfem2D_fh_config INTERFACE ${CMAKE_CURRENT_BINARY_DIR}/config.fh)
add_library(meshfem2D::fh_config ALIAS meshfem2D_fh_config)
target_include_directories(meshfem2D_fh_config INTERFACE ${CMAKE_CURRENT_BINARY_DIR})
