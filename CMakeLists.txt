cmake_minimum_required(VERSION 3.17.5)

project(SPECFEMPP VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
option(HDF5_CXX_BUILD "Build HDF5 C++" ON)
option(VTK_CXX_BUILD "Build VTK C++" ON)
option(MPI_PARALLEL "MPI enabled" OFF)
option(BUILD_TESTS "Tests included" OFF)
option(BUILD_EXAMPLES "Examples included" ON)
option(ENABLE_SIMD "Enable SIMD" OFF)
option(ENABLE_PROFILING "Enable profiling" OFF)
option(SPECFEMPP_ENABLE_PYTHON "Enable Python binding" OFF)
# set(CMAKE_BUILD_TYPE Release)
set(CHUNK_SIZE 32)
set(NUM_CHUNKS 1)
set(NUM_THREADS 160)
set(NUM_VECTOR_LANES 1)

if (SPECFEMPP_ENABLE_PYTHON)
    message("-- Adding -fPIC flag for Python binding.")
    if(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
            set(CMAKE_CXX_FLAGS "-fp-model=precise -fPIC")
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
            message("-- Detected Intel classic compiler which will be deprecated soon.")
            message("-- It is recommended you use IntelLLVM compiler.")
            set(CMAKE_CXX_FLAGS "-diag-disable=10441 -fp-model=precise -fPIC")
    else()
            set(CMAKE_CXX_FLAGS "-fPIC")
    endif()
else (SPECFEMPP_ENABLE_PYTHON)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
            set(CMAKE_CXX_FLAGS "-fp-model=precise")
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
            message("-- Detected Intel classic compiler which will be deprecated soon.")
            message("-- It is recommended you use IntelLLVM compiler.")
            set(CMAKE_CXX_FLAGS "-diag-disable=10441 -fp-model=precise")
    endif()
endif (SPECFEMPP_ENABLE_PYTHON)

# Check if MacOS
if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(__APPLE__ TRUE)
    message("-- macOS detected -- setting __APPLE__ TRUE")
else(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(__APPLE__ FALSE)
    message("-- macOS not detected -- setting __APPLE__ FALSE")
endif()

if (DEFINED Kokkos_ENABLE_CUDA)
        if (Kokkos_ENABLE_CUDA)
                # message("Setting CUDA variables")
                set(Kokkos_ENABLE_CUDA_LAMBDA ON CACHE BOOL "Using CUDA Lambda by default")
                set(Kokkos_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE ON CACHE BOOL "Using CUDA Relocatable device by default")
        endif()
endif()

# Install Kokkos as a dependency
## TODO: Add options for on utilizing in house builds
include(FetchContent)
FetchContent_Declare(
kokkos
URL https://github.com/kokkos/kokkos/archive/refs/tags/4.3.01.zip
)
FetchContent_MakeAvailable(kokkos)

FetchContent_Declare(
        yaml
        URL https://github.com/jbeder/yaml-cpp/archive/refs/tags/0.8.0.tar.gz
)
FetchContent_MakeAvailable(yaml)

include_directories(BEFORE SYSTEM ${yaml_BINARY_DIR} ${yaml_SOURCE_DIR}/include)

find_package(VTK COMPONENTS
CommonColor
CommonCore
FiltersSources
InteractionStyle
RenderingContextOpenGL2
RenderingCore
RenderingFreeType
RenderingGL2PSOpenGL2
RenderingOpenGL2
)


if (NOT VTK_FOUND)
  message("VTK not found: ${VTK_NOT_FOUND_MESSAGE}")
  set(VTK_CXX_BUILD OFF)
else ()
  message(STATUS "    VTK:   ${VTK_LIBRARIES}")
endif()


# Try finding boost and if not found install.
find_package(Boost 1.73.0 COMPONENTS program_options filesystem system)

if (NOT ${Boost_FOUND})
        add_subdirectory(boost-cmake)
else ()
        message(STATUS "    LIB:   ${Boost_LIBRARY_DIRS}")
        message(STATUS "    INC:   ${Boost_INCLUDE_DIRS}")
        message(STATUS "    LIBSO: ${Boost_LIBRARIES}")
endif()

# Install HDF5 as a dependency if not found
find_package(HDF5 COMPONENTS CXX)

if (NOT ${HDF5_FOUND})
        message("-- HDF5 not found. Building without HDF5.")
        set(HDF5_CXX_BUILD OFF)
else ()
        message(STATUS "    LIB:   ${HDF5_LIBRARIES}")
        message(STATUS "    INC:   ${HDF5_INCLUDE_DIRS}")
        message(STATUS "    LIBSO: ${HDF5_CXX_LIBRARIES}")
endif()

configure_file(constants.hpp.in constants.hpp)

include_directories(include)
include_directories(${CMAKE_BINARY_DIR})

add_subdirectory(meshfem2d)

if (ENABLE_SIMD)
        message("-- Enabling SIMD")
        add_definitions(-DENABLE_SIMD)
endif()

if (ENABLE_PROFILING)
        message("-- Enabling profiling")
        add_definitions(-DENABLE_PROFILING)
endif()

# Build specfem2d libraries
add_library(
        quadrature
        src/quadrature/quadrature.cpp
        src/quadrature/gll/gll_utils.cpp
        src/quadrature/gll/gll_library.cpp
        src/quadrature/gll/lagrange_poly.cpp
        src/quadrature/gll/gll.cpp
)

target_link_libraries(
        quadrature
        Kokkos::kokkos
)

add_library(
        IO
        src/IO/fortranio/fortran_io.cpp
        src/IO/HDF5/native_type.cpp
        # src/IO/ASCII/native_type.cpp
        src/IO/sources.cpp
        src/IO/receivers.cpp
        src/IO/mesh.cpp
        src/IO/mesh/impl/fortran/read_boundaries.cpp
        src/IO/mesh/impl/fortran/read_elements.cpp
        src/IO/mesh/impl/fortran/read_material_properties.cpp
        src/IO/mesh/impl/fortran/read_mesh_database.cpp
        src/IO/mesh/impl/fortran/read_interfaces.cpp
        src/IO/mesh/impl/fortran/read_parameters.cpp
)

if (NOT HDF5_CXX_BUILD)
        target_compile_definitions(
                IO
                PUBLIC -DNO_HDF5
        )

        target_link_libraries(
                IO
                Boost::boost
                Kokkos::kokkos
        )
else()
        target_link_libraries(
                IO
                ${HDF5_LIBRARIES}
                Boost::boost
                Kokkos::kokkos
        )
endif()

target_link_libraries(
  IO
  mesh
  source_class
  receiver_class
)

add_library(
        point
        src/point/coordinates.cpp
        src/point/partial_derivatives.cpp
)

target_link_libraries(
        point
        Kokkos::kokkos
)

add_library(
        enumerations
        src/enumerations/medium.cpp
)

add_library(
        edge
        src/edge/interface.cpp
)

target_link_libraries(
        edge
        Kokkos::kokkos
)

add_library(
        specfem_mpi
        src/specfem_mpi/specfem_mpi.cpp
)

if (MPI_PARALLEL)
        target_compile_definitions(
                specfem_mpi
                PUBLIC -DMPI_PARALLEL
        )
        message("-- Compiling SPECFEM with MPI")
else()
        message("-- Compiling SPECFEM without MPI")
endif(MPI_PARALLEL)

# add_library(
#         material_class
#         src/material/elastic_isotropic_material.cpp
#         src/material/acoustic_isotropic_material.cpp
# )

# target_link_libraries(
#         material_class
#         Kokkos::kokkos
#         specfem_mpi
# )

add_library(
        mesh
        src/mesh/boundaries/forcing_boundaries.cpp
        src/mesh/boundaries/absorbing_boundaries.cpp
        src/mesh/boundaries/acoustic_free_surface.cpp
        src/mesh/elements/tangential_elements.cpp
        src/mesh/elements/axial_elements.cpp
        # src/mesh/mpi_interfaces/mpi_interfaces.cpp
        src/mesh/materials/materials.cpp
        src/mesh/coupled_interfaces/interface_container.cpp
        src/mesh/coupled_interfaces/coupled_interfaces.cpp
        src/mesh/tags/tags.cpp
        src/mesh/mesh.cpp
)

target_link_libraries(
        mesh
        Kokkos::kokkos
        specfem_mpi
        # material_class
        yaml-cpp
)

add_library(
        jacobian
        src/jacobian/shape_functions.cpp
        src/jacobian/jacobian.cpp
)

target_link_libraries(
        jacobian
        Kokkos::kokkos
        point
)

# add_library(
#         utilities
#         src/utilities/utilities.cpp
# )

# target_link_libraries(
#         utilities
#         jacobian
#         Kokkos::kokkos
#         specfem_mpi
# )

add_library(
        read_seismogram
        src/reader/seismogram.cpp
)

target_link_libraries(
        read_seismogram
        Kokkos::kokkos
        )

add_library(
        reader
        src/reader/wavefield.cpp
)

target_link_libraries(
        reader
        compute
        IO
        read_seismogram
)

add_library(
        algorithms
        src/algorithms/locate_point.cpp
)

target_link_libraries(
        algorithms
        Kokkos::kokkos
        jacobian
        point
)

add_library(
        source_time_function
        src/source_time_function/dirac.cpp
        src/source_time_function/ricker.cpp
        src/source_time_function/external.cpp
)

target_link_libraries(
        source_time_function
        read_seismogram
        Kokkos::kokkos
        point
)

add_library(
        source_class
        src/source/source.cpp
        src/source/force_source.cpp
        src/source/moment_tensor_source.cpp
        src/source/adjoint_source.cpp
        src/source/external.cpp
)

target_link_libraries(
        source_class
        Kokkos::kokkos
        specfem_mpi
        # utilities
        quadrature
        source_time_function
        yaml-cpp
        point
        algorithms
        Boost::boost
)

add_library(
        receiver_class
        src/receiver/receiver.cpp
)

target_link_libraries(
        receiver_class
        specfem_mpi
        Kokkos::kokkos
        # utilities
        quadrature
        yaml-cpp
        Boost::boost
)

add_library(
        compute
        src/compute/compute_mesh.cpp
        src/compute/compute_partial_derivatives.cpp
        src/compute/compute_properties.cpp
        src/compute/compute_kernels.cpp
        src/compute/compute_sources.cpp
        src/compute/compute_receivers.cpp
        src/compute/coupled_interfaces.cpp
        src/compute/boundaries/impl/acoustic_free_surface.cpp
        src/compute/boundaries/impl/stacey.cpp
        src/compute/boundaries/boundaries.cpp
        src/compute/fields/fields.cpp
        src/compute/compute_boundary_values.cpp
        src/compute/assembly/assembly.cpp
        src/compute/assembly/compute_wavefield.cpp
)

target_link_libraries(
        compute
        quadrature
        mesh
        # material_class
        source_class
        jacobian
        point
        edge
        receiver_class
        Kokkos::kokkos
)

add_library(
        domain
        src/domain/impl/boundary_conditions/none/none.cpp
        src/domain/impl/boundary_conditions/none/print.cpp
        src/domain/impl/boundary_conditions/dirichlet/dirichlet.cpp
        src/domain/impl/boundary_conditions/dirichlet/print.cpp
        src/domain/impl/boundary_conditions/stacey/stacey.cpp
        src/domain/impl/boundary_conditions/stacey/mass_matrix.cpp
        src/domain/impl/boundary_conditions/stacey/print.cpp
        src/domain/impl/boundary_conditions/composite_stacey_dirichlet/composite_stacey_dirichlet.cpp
        src/domain/impl/boundary_conditions/composite_stacey_dirichlet/mass_matrix.cpp
        src/domain/impl/boundary_conditions/composite_stacey_dirichlet/print.cpp
        src/domain/impl/boundary_conditions/boundary_conditions.cpp
        src/domain/impl/elements/acoustic/acoustic2d.cpp
        src/domain/impl/elements/elastic/elastic2d.cpp
        src/domain/impl/elements/element.cpp
        src/domain/impl/elements/kernel.cpp
        src/domain/impl/kernels.cpp
        src/domain/domain.cpp
)

target_link_libraries(
        domain
        Kokkos::kokkos
        enumerations
)

add_library(coupled_interface
        src/coupled_interface/coupled_interface.cpp
)

target_link_libraries(
        coupled_interface
        Kokkos::kokkos
        compute
)

# add_library(
#         kernels
#         src/kernels/kernels.cpp
# )

# target_link_libraries(
#         kernels
#         Kokkos::kokkos
#         compute
#         coupled_interface
# )

add_library(
        frechet_derivatives
        src/frechet_derivatives/impl/element_kernel/elastic_isotropic.cpp
        src/frechet_derivatives/impl/element_kernel/acoustic_isotropic.cpp
        src/frechet_derivatives/impl/frechet_element.cpp
        src/frechet_derivatives/frechet_derivatives.cpp
)

target_link_libraries(
        frechet_derivatives
        Kokkos::kokkos
        point
)

add_library(
        kernels
        src/kernels/frechet_kernels.cpp
)

target_link_libraries(
        kernels
        Kokkos::kokkos
        compute
        frechet_derivatives
)

add_library(
        timescheme
        src/timescheme/timescheme.cpp
        src/timescheme/newmark.cpp
)

target_link_libraries(
        timescheme
        Kokkos::kokkos
        yaml-cpp
        compute
)

add_library(
        solver
        src/solver/time_marching.cpp
)

target_link_libraries(
        solver
        Kokkos::kokkos
        timescheme
        domain
)

add_library(
        writer
        src/writer/seismogram.cpp
        src/writer/wavefield.cpp
        src/writer/kernel.cpp
)

target_link_libraries(
        writer
        compute
        receiver_class
        IO
)

add_library(
        plotter
        src/plotter/plot_wavefield.cpp
)

if (NOT VTK_CXX_BUILD)
        target_compile_definitions(
                plotter
                PUBLIC -DNO_VTK
        )

        target_link_libraries(
                plotter
                compute
        )
else ()
        target_link_libraries(
                plotter
                compute
                ${VTK_LIBRARIES}
                )

        # Only define the __APPLE__ compile definition if it is defined
        if (__APPLE__)
        target_compile_definitions(plotter PRIVATE __APPLE__)
        endif(__APPLE__)
endif()

add_library(
        parameter_reader
        src/parameter_parser/run_setup.cpp
        # src/parameter_parser/solver/solver.cpp
        src/parameter_parser/time_scheme/time_scheme.cpp
        src/parameter_parser/database_configuration.cpp
        src/parameter_parser/header.cpp
        src/parameter_parser/quadrature.cpp
        src/parameter_parser/receivers.cpp
        src/parameter_parser/writer/seismogram.cpp
        src/parameter_parser/setup.cpp
        src/parameter_parser/writer/wavefield.cpp
        src/parameter_parser/writer/plot_wavefield.cpp
        src/parameter_parser/writer/kernel.cpp
)

target_link_libraries(
        parameter_reader
        quadrature
        timescheme
        receiver_class
        yaml-cpp
        writer
        reader
        kernels
        domain
        solver
        Boost::filesystem
)

add_executable(
        specfem2d
        src/specfem2d.cpp
)



target_link_libraries(
        specfem2d
        specfem_mpi
        Kokkos::kokkos
        mesh
        quadrature
        compute
        source_class
        parameter_reader
        receiver_class
        writer
        plotter
        reader
        domain
        coupled_interface
        kernels
        solver
        Boost::program_options
)



# add_library(
#         event_marching
# )
# target_link_libraries(
#         event_marching
#         quadrature
#         kernels
#         domain
#         solver
# )

add_library(
        _util
        src/_util/build_demo_assembly.cpp
)

target_link_libraries(
        _util
        quadrature
        IO
        point
        edge
        specfem_mpi
        mesh
        jacobian
        reader
        algorithms
        source_time_function
        source_class
        receiver_class
        compute
        domain
        coupled_interface
        frechet_derivatives
        kernels
        timescheme
        solver
        writer
        yaml-cpp
        Boost::filesystem
)


add_library(
        adjacency_graph
        src/adjacency_graph/adjacency_graph.cpp
)

target_link_libraries(
        adjacency_graph
        mesh
        compute
        domain
        kernels
)

add_executable(
        specfem2d_eventmarcher
        src/specfem2d_eventmarcher.cpp
)


#TODO verify which libs we actually need.
target_link_libraries(
        specfem2d_eventmarcher
        adjacency_graph
        # event_marching
        specfem_mpi
        Kokkos::kokkos
        mesh
        quadrature
        compute
        source_class
        parameter_reader
        receiver_class
        writer
        plotter
        reader
        coupled_interface
        frechet_derivatives
        timescheme
        kernels
        domain
        solver
        Boost::program_options
        _util
        IO
)

# Include tests
if (BUILD_TESTS)
        message("-- Including tests.")
        add_subdirectory(tests/unit-tests)
endif()

if (BUILD_EXAMPLES)
        message("-- Including examples.")
        add_subdirectory(examples)
endif()

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

if (SPECFEMPP_USE_SKBUILD AND EXISTS ${SKBUILD_SCRIPTS_DIR})
    install(TARGETS specfem2d DESTINATION ${SKBUILD_SCRIPTS_DIR})
    install(FILES ${CMAKE_BINARY_DIR}/xmeshfem2D DESTINATION ${SKBUILD_SCRIPTS_DIR} PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE)
endif ()

if (SPECFEMPP_ENABLE_PYTHON)
    find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
    find_package(pybind11 CONFIG REQUIRED)
    python_add_library(_core MODULE src/specfem2d.cpp WITH_SOABI)
    target_link_libraries(
        _core PRIVATE
        specfem_mpi
        Kokkos::kokkos
        mesh
        quadrature
        compute
        source_class
        parameter_reader
        receiver_class
        writer
        reader
        domain
        coupled_interface
        kernels
        solver
        Boost::program_options
        pybind11::headers
    )
    target_compile_definitions(_core PRIVATE SPECFEMPP_ENABLE_PYTHON)
    target_compile_definitions(_core PRIVATE VERSION_INFO=${PROJECT_VERSION})
    install(TARGETS _core DESTINATION specfempp_core)
endif (SPECFEMPP_ENABLE_PYTHON)
