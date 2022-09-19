cmake_minimum_required(VERSION 3.17.5)

project(specfem2d_kokkos VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)

# Install Kokkos as a dependency
## TODO: Add options for on utilizing in house builds
include(FetchContent)
FetchContent_Declare(
  kokkos
  URL https://github.com/kokkos/kokkos/archive/refs/tags/3.6.01.zip
)
FetchContent_MakeAvailable(kokkos)


# Build specfem2d libraries
## TODO: Reorgnanize into separate directories
add_library(
        gll_utils
        src/gll_utils.cpp
        include/gll_utils.h
)

target_link_libraries(
        gll_utils PUBLIC
        Kokkos::kokkos
)

add_library(
        gll_library
        src/gll_library.cpp
        include/gll_library.h
)

target_link_libraries(
        gll_library
        gll_utils
)

add_library(
        lagrange
        src/lagrange_poly.cpp
        include/lagrange_poly.h
)

target_link_libraries(
        lagrange
        gll_library
        Kokkos::kokkos
)

add_library(
        quadrature
        src/quadrature.cpp
        include/quadrature.h
        include/kokkos_abstractions.h
)

target_link_libraries(
        quadrature
        lagrange
        gll_library
        Kokkos::kokkos
)

# Include tests
add_subdirectory(tests/unittests)

# Doxygen

# look for Doxygen package
find_package(Doxygen)

if (DOXYGEN_FOUND)
    # set input and output files
    set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in)
    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile.out)

    # request to configure the file
    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
    message("Doxygen build started")

    # Note: do not put "ALL" - this builds docs together with application EVERY TIME!
    add_custom_target( docs
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM )
else (DOXYGEN_FOUND)
  message("Doxygen need to be installed to generate the doxygen documentation")
endif (DOXYGEN_FOUND)
