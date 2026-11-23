# Injection points: builds the points handed to an injection plugin from the
# assembled injection surface. Sits on top of the assembly and the provider
# session, which itself stays free of meshes and assemblies.
file(GLOB_RECURSE INJECTION_POINTS_SOURCES "*.cpp")
add_library(specfem_injection_points STATIC ${INJECTION_POINTS_SOURCES})
add_library(specfem::injection_points ALIAS specfem_injection_points)

target_include_directories(specfem_injection_points
    PUBLIC
        ${CMAKE_SOURCE_DIR}/core
)

target_link_libraries(specfem_injection_points
    PUBLIC
        specfem::injection_session
        specfem::assembly
        specfem::mpi
        Kokkos::kokkos
        $<TARGET_NAME_IF_EXISTS:MPI::MPI_CXX>
    PRIVATE
        specfem::element
)
