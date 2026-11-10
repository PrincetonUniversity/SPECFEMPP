# Injection provider session: drives a loaded plugin through its lifecycle and
# keeps a device-resident window of the incident field. This is the SPECFEM++
# side of the plugin ABI, so it may use Kokkos and MPI; plugins never see it.
file(GLOB_RECURSE INJECTION_SESSION_SOURCES "*.cpp")
add_library(specfem_injection_session STATIC ${INJECTION_SESSION_SOURCES})
add_library(specfem::injection_session ALIAS specfem_injection_session)

target_include_directories(specfem_injection_session
    PUBLIC
        ${CMAKE_SOURCE_DIR}/core
)

target_link_libraries(specfem_injection_session
    PUBLIC
        specfem::injection
        specfem::mpi
        Kokkos::kokkos
        $<TARGET_NAME_IF_EXISTS:MPI::MPI_CXX>
)
