# Add existing subdirectories
add_subdirectory(assembly)
add_subdirectory(point)
add_subdirectory(receivers)
add_subdirectory(shape_functions)
add_subdirectory(source)
add_subdirectory(jacobian)

# Core library - singleton for managing initialization/finalization
add_library(specfem_core STATIC core.cpp)
target_include_directories(specfem_core PUBLIC ${CMAKE_SOURCE_DIR}/core)
target_link_libraries(specfem_core
    PUBLIC
        Kokkos::kokkos
        yaml-cpp::yaml-cpp
        ${BOOST_LIBS}
        # Add MPI dependency when available
        $<TARGET_NAME_IF_EXISTS:MPI::MPI_CXX>
)

# Execute library - main simulation execution logic
add_library(specfem_execute STATIC execute.cpp)
target_include_directories(specfem_execute PUBLIC ${CMAKE_SOURCE_DIR}/core)
target_link_libraries(specfem_execute
    PUBLIC
        specfem_core
        yaml-cpp::yaml-cpp
        ${BOOST_LIBS}
        # All the dependencies from the original execute library
        specfem_mpi
        Kokkos::kokkos
        mesh
        quadrature
        assembly
        source_class
        parameter_reader
        specfem::receivers
        writer
        periodic_tasks
        reader
        coupled_interface
        kokkos_kernels
        solver
        ${BOOST_LIBS}
)

# Create an interface library to group all specfem components
add_library(specfem_unified INTERFACE)
target_link_libraries(specfem_unified
    INTERFACE
        specfem_core
        specfem_execute
)
