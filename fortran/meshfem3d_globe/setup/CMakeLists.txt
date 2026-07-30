project(MESHFEM3DG_SETUP LANGUAGES Fortran C)
cmake_minimum_required(VERSION 3.10)

message(STATUS "-- Configuring meshfem3Dg headers...")

enable_language(Fortran)
enable_language(C)

set(CMAKE_Fortran_STANDARD 95)
set(CMAKE_Fortran_STANDARD_REQUIRED ON)

include_directories(${CMAKE_CURRENT_BINARY_DIR})

if(BEOWULF_CLUSTER)
  message(STATUS "-- Building for Beowulf cluster")
  set(LOCAL_PATH_IS_ALSO_GLOBAL "false")
else()
  message(STATUS "-- Building for shared memory machine")
  set(LOCAL_PATH_IS_ALSO_GLOBAL "true")
endif()

include(CheckIncludeFile)
include(CheckFunctionExists)
include(CheckSymbolExists)
include(CheckCSourceCompiles)
include(CheckCCompilerFlag)
include(CheckFortranCompilerFlag)
include(CheckLibraryExists)

check_include_file(err.h HAVE_ERR)
check_include_file(inttypes.h HAVE_INTTYPES_H)
check_include_file(stdint.h HAVE_STDINT_H)
check_include_file(stdio.h HAVE_STDIO_H)
check_include_file(stdlib.h HAVE_STDLIB_H)
check_include_file(strings.h HAVE_STRINGS_H)
check_include_file(string.h HAVE_STRING_H)
check_include_file(sys/stat.h HAVE_SYS_STAT_H)
check_include_file(sys/types.h HAVE_SYS_TYPES_H)
check_include_file(unistd.h HAVE_UNISTD_H)
check_include_file(emmintrin.h HAVE_EMMINTRIN)
check_include_file(xmmintrin.h HAVE_XMMINTRIN)

find_package(Threads REQUIRED)
if(Threads_FOUND)
  set(HAVE_PTHREAD 1)
endif()

option(USE_MAP_FUNCTION "Enable optimized file I/O for regional simulations" OFF)
if(USE_MAP_FUNCTION)
  set(USE_MAP_FUNCTION 1)
endif()

set(PACKAGE_NAME "SPECFEM3D_GLOBE")
set(PACKAGE_VERSION "8.0.0")
set(PACKAGE_STRING "${PACKAGE_NAME} ${PACKAGE_VERSION}")
set(PACKAGE_TARNAME "specfem3d_globe")
set(PACKAGE_URL "https://github.com/geodynamics/specfem3d_globe")
set(PACKAGE_BUGREPORT "support@specfem3d.org")

set(GIT_PACKAGE_VERSION "${SPECFEMPP_GIT_VERSION}")
set(GIT_COMMIT_VERSION "${SPECFEMPP_GIT_COMMIT_HASH}")
set(GIT_DATE_VERSION "${SPECFEMPP_GIT_COMMIT_DATE}")

check_c_source_compiles("
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
extern char *yytext;
int main() { return 0; }
" YYTEXT_POINTER)

if(YYTEXT_POINTER)
  set(YYTEXT_POINTER 1)
else()
  set(YYTEXT_POINTER 0)
endif()

include(FortranCInterface)
FortranCInterface_VERIFY()

set(FC_FUNC "")
if(FortranCInterface_GLOBAL_CASE STREQUAL "UPPER")
  string(APPEND FC_FUNC "${FortranCInterface_GLOBAL_PREFIX}NAME##${FortranCInterface_GLOBAL_MIDDLE}${FortranCInterface_GLOBAL_SUFFIX}")
else()
  string(APPEND FC_FUNC "${FortranCInterface_GLOBAL_PREFIX}name##${FortranCInterface_GLOBAL_MIDDLE}${FortranCInterface_GLOBAL_SUFFIX}")
endif()

set(FC_FUNC_ "")
if(FortranCInterface_GLOBAL__CASE STREQUAL "UPPER")
  string(APPEND FC_FUNC_ "${FortranCInterface_GLOBAL__PREFIX}NAME##${FortranCInterface_GLOBAL__MIDDLE}${FortranCInterface_GLOBAL__SUFFIX}")
else()
  string(APPEND FC_FUNC_ "${FortranCInterface_GLOBAL__PREFIX}name##${FortranCInterface_GLOBAL__MIDDLE}${FortranCInterface_GLOBAL__SUFFIX}")
endif()

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/config.h.in
               ${CMAKE_CURRENT_BINARY_DIR}/config.h @ONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/config.fh.in
               ${CMAKE_CURRENT_BINARY_DIR}/config.fh @ONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/constants.h.in
               ${CMAKE_CURRENT_BINARY_DIR}/constants.h @ONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/precision.h.in
               ${CMAKE_CURRENT_BINARY_DIR}/precision.h @ONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/version.fh.in
               ${CMAKE_CURRENT_BINARY_DIR}/version.fh @ONLY)

add_library(meshfem3Dg_config INTERFACE ${CMAKE_CURRENT_BINARY_DIR}/config.h)
add_library(meshfem3Dg::config ALIAS meshfem3Dg_config)
target_include_directories(meshfem3Dg_config INTERFACE ${CMAKE_CURRENT_BINARY_DIR})

add_library(meshfem3Dg_config_fortran INTERFACE ${CMAKE_CURRENT_BINARY_DIR}/config.fh)
add_library(meshfem3Dg::config_fortran ALIAS meshfem3Dg_config_fortran)
target_include_directories(meshfem3Dg_config_fortran INTERFACE ${CMAKE_CURRENT_BINARY_DIR})

add_library(meshfem3Dg_constants INTERFACE ${CMAKE_CURRENT_BINARY_DIR}/constants.h)
add_library(meshfem3Dg::constants ALIAS meshfem3Dg_constants)
target_include_directories(meshfem3Dg_constants INTERFACE ${CMAKE_CURRENT_BINARY_DIR})

add_library(meshfem3Dg_precision INTERFACE ${CMAKE_CURRENT_BINARY_DIR}/precision.h)
add_library(meshfem3Dg::precision ALIAS meshfem3Dg_precision)
target_include_directories(meshfem3Dg_precision INTERFACE ${CMAKE_CURRENT_BINARY_DIR})

add_library(meshfem3Dg_version INTERFACE ${CMAKE_CURRENT_BINARY_DIR}/version.fh)
add_library(meshfem3Dg::version ALIAS meshfem3Dg_version)
target_include_directories(meshfem3Dg_version INTERFACE ${CMAKE_CURRENT_BINARY_DIR})
