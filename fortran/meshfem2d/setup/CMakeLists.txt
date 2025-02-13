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

# ====== FIGURE OUT THE FORTRAN NAME MANGLING END =============

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
