# Project name and language
project(MESHFEM3D_SETUP LANGUAGES Fortran C)
cmake_minimum_required(VERSION 3.10)

message(STATUS "-- Configuring meshfem3D headers...")

enable_language(Fortran)
enable_language(C)

# Set the Fortran standard to 95
set(CMAKE_Fortran_STANDARD 95)
set(CMAKE_Fortran_STANDARD_REQUIRED ON)

# Set include directories
include_directories(${CMAKE_CURRENT_BINARY_DIR})

if (BEOWULF_CLUSTER)
    message(STATUS "-- Building for Beowulf cluster")
    set(LOCAL_PATH_IS_ALSO_GLOBAL "false")
else()
    message(STATUS "-- Building for shared memory machine")
    set(LOCAL_PATH_IS_ALSO_GLOBAL "true")
endif()

# ================== Figuring out available headers START ================
include(CheckIncludeFile)
include(CheckFunctionExists)
include(CheckSymbolExists)
include(CheckCSourceCompiles)
include(CheckCCompilerFlag)
include(CheckFortranCompilerFlag)
include(CheckLibraryExists)

# Check for required headers
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

# Check for compiler-specific headers
include(CheckIncludeFile)
check_include_file(emmintrin.h HAVE_EMMINTRIN)
check_include_file(xmmintrin.h HAVE_XMMINTRIN)

# Check for libraries
include(CheckLibraryExists)
check_symbol_exists(vtkCommon "vtkCommon.h" HAVE_LIBVTKCOMMON)
check_symbol_exists(vtkDICOMParser "vtkDICOMParser.h" HAVE_LIBVTKDICOMPARSER)
check_symbol_exists(vtkexpat "vtkexpat.h" HAVE_LIBVTKEXPAT)
check_symbol_exists(vtkFiltering "vtkFiltering.h" HAVE_LIBVTKFILTERING)
check_symbol_exists(vtkGenericFiltering "vtkGenericFiltering.h" HAVE_LIBVTKGENERICFILTERING)
check_symbol_exists(vtkGraphics "vtkGraphics.h" HAVE_LIBVTKGRAPHICS)
check_symbol_exists(vtkRendering "vtkRendering.h" HAVE_LIBVTKRENDERING)
check_symbol_exists(vtksys "vtksys.h" HAVE_LIBVTKSYS)
check_symbol_exists(vtkzlib "vtkzlib.h" HAVE_LIBVTKZLIB)

# Check for threading support
find_package(Threads REQUIRED)
if(Threads_FOUND)
    set(HAVE_PTHREAD 1)
endif()


# Set up definition for use in CMake
option(USE_MAP_FUNCTION "Enable optimized file I/O for regional simulations" OFF)
if(USE_MAP_FUNCTION)
    set(USE_MAP_FUNCTION 1)
endif()

# ================== Figuring out available headers END ================

# Package variables
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
set(SPECFEM2D_RELEASE_VERSION 0) # Set to 1 if it's a stable release

set(GIT_PACKAGE_VERSION "v4.1.1-100-g725321e6")
set(GIT_COMMIT_VERSION "725321e6599f90e486c907435840df559ebc0a9c")
set(GIT_DATE_VERSION "2024-11-21 11:50:38 +0100")


# =============== CHECK if the compiler supports yytext START =========
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
# =============== CHECK if the compiler supports yytext END ============


# =============== FIGURING OUT FORTRAN NAME MANGLING START =============
# Write test Fortran subroutines
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/test.f90" "
subroutine foobar()
return
end
subroutine foo_bar()
return
end
")

# Test combinations of case, underscore, and extra underscore
foreach(TEST_CASE
   "lower;no;no"      # lower case, no underscore, no extra underscore
   "lower;no;yes"     # lower case, no underscore, extra underscore
   "lower;yes;no"     # lower case, underscore, no extra underscore
   "lower;yes;yes"    # lower case, underscore, extra underscore
   "upper;no;no"      # upper case, no underscore, no extra underscore
   "upper;no;yes"     # upper case, no underscore, extra underscore
   "upper;yes;no"     # upper case, underscore, no extra underscore
   "upper;yes;yes"    # upper case, underscore, extra underscore
)
   # Parse the test case
   list(GET TEST_CASE 0 CASE)
   list(GET TEST_CASE 1 HAS_UNDERSCORE)
   list(GET TEST_CASE 2 HAS_EXTRA_UNDERSCORE)

   # Construct names to test based on case
   if(CASE STREQUAL "lower")
       set(NAME1 "foobar")
       set(NAME2 "foo_bar")
   else()
       set(NAME1 "FOOBAR")
       set(NAME2 "FOO_BAR")
   endif()

   # Add underscore if needed
   if(HAS_UNDERSCORE STREQUAL "yes")
       set(NAME1 "${NAME1}_")
       set(NAME2 "${NAME2}_")
   endif()

   # Add extra underscore to NAME2 if needed
   if(HAS_EXTRA_UNDERSCORE STREQUAL "yes")
       set(NAME2 "${NAME2}_")
   endif()

   # Write C test file
   file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/test.c" "
   extern void ${NAME1}(void);
   extern void ${NAME2}(void);
   int main() {
       ${NAME1}();
       ${NAME2}();
       return 0;
   }
   ")

   try_compile(COMPILE_SUCCESS
       ${CMAKE_CURRENT_BINARY_DIR}
       SOURCES
           "${CMAKE_CURRENT_BINARY_DIR}/test.f90"
           "${CMAKE_CURRENT_BINARY_DIR}/test.c"
       OUTPUT_VARIABLE COMPILE_OUTPUT
   )

   if(COMPILE_SUCCESS)
       set(FC_CASE ${CASE} CACHE STRING "Fortran name case")
       set(FC_UNDERSCORE ${HAS_UNDERSCORE} CACHE STRING "Fortran name underscore")
       set(FC_EXTRA_UNDERSCORE ${HAS_EXTRA_UNDERSCORE} CACHE STRING "Fortran name extra underscore")

       # Set the mangling description string like in the autoconf file
       set(FC_MANGLING "${CASE} case, ${HAS_UNDERSCORE} underscore, ${HAS_EXTRA_UNDERSCORE} extra underscore"
           CACHE STRING "Fortran name mangling scheme")

       message(STATUS "Detected Fortran name mangling: ${FC_MANGLING}")
       break()
   endif()
endforeach()

if(NOT DEFINED FC_MANGLING)
   message(FATAL_ERROR "Could not determine Fortran name mangling")
endif()

# Now define the FC_FUNC and FC_FUNC_ macros based on the detected scheme
if(FC_CASE STREQUAL "lower")
   if(FC_UNDERSCORE STREQUAL "no")
       if(FC_EXTRA_UNDERSCORE STREQUAL "no")
           set(FC_FUNC "name")
           set(FC_FUNC_ "name")
       else()
           set(FC_FUNC "name")
           set(FC_FUNC_ "name ## _")
       endif()
   else()
       if(FC_EXTRA_UNDERSCORE STREQUAL "no")
           set(FC_FUNC "name ## _")
           set(FC_FUNC_ "name ## _")
       else()
           set(FC_FUNC "name ## _")
           set(FC_FUNC_ "name ## __")
       endif()
   endif()
else()
   if(FC_UNDERSCORE STREQUAL "no")
       if(FC_EXTRA_UNDERSCORE STREQUAL "no")
           set(FC_FUNC "NAME")
           set(FC_FUNC_ "NAME")
       else()
           set(FC_FUNC "NAME")
           set(FC_FUNC_ "NAME ## _")
       endif()
   else()
       if(FC_EXTRA_UNDERSCORE STREQUAL "no")
           set(FC_FUNC "NAME ## _")
           set(FC_FUNC_ "NAME ## _")
       else()
           set(FC_FUNC "NAME ## _")
           set(FC_FUNC_ "NAME ## __")
       endif()
   endif()
endif()

# =============== FIGURING OUT FORTRAN NAME MANGLING END ===============


# ===================== config.h.in START ==================
# Generate config.h file
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/config.h.in
               ${CMAKE_CURRENT_BINARY_DIR}/config.h @ONLY)

# ===================== config.h.in END =====================

# ===================== config.fh.in START ==================

set(CONFIGURE_FLAGS "FC=${CMAKE_Fortran_COMPILER_ID} FCFLAGS=${CMAKE_Fortran_FLAGS}")

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
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/version.fh.in
               ${CMAKE_CURRENT_BINARY_DIR}/version.fh @ONLY)
# ===================== version.h.in START ================

add_library(meshfem3D_config INTERFACE ${CMAKE_CURRENT_BINARY_DIR}/config.h)
add_library(meshfem3D::config ALIAS meshfem3D_config)
target_include_directories(meshfem3D_config INTERFACE ${CMAKE_CURRENT_BINARY_DIR})

add_library(meshfem3D_config_fortran INTERFACE ${CMAKE_CURRENT_BINARY_DIR}/config.fh)
add_library(meshfem3D::config_fortran ALIAS meshfem3D_config_fortran)
target_include_directories(meshfem3D_config_fortran INTERFACE ${CMAKE_CURRENT_BINARY_DIR})

add_library(meshfem3D_constants INTERFACE ${CMAKE_CURRENT_BINARY_DIR}/constants.h)
add_library(meshfem3D::constants ALIAS meshfem3D_constants)
target_include_directories(meshfem3D_constants INTERFACE ${CMAKE_CURRENT_BINARY_DIR})

add_library(meshfem3D_precision INTERFACE ${CMAKE_CURRENT_BINARY_DIR}/precision.h)
add_library(meshfem3D::precision ALIAS meshfem3D_precision)
target_include_directories(meshfem3D_precision INTERFACE ${CMAKE_CURRENT_BINARY_DIR})

add_library(meshfem3D_version INTERFACE ${CMAKE_CURRENT_BINARY_DIR}/version.fh)
add_library(meshfem3D::version ALIAS meshfem3D_version)
target_include_directories(meshfem3D_version INTERFACE ${CMAKE_CURRENT_BINARY_DIR})
