file(GLOB_RECURSE LINEAR_SYSTEM_SOURCE_FILES "./*.cpp" "./**/*.cpp")

add_library(specfem_linear_system ${LINEAR_SYSTEM_SOURCE_FILES})
add_library(specfem::linear_system ALIAS specfem_linear_system)

target_link_libraries(
        specfem_linear_system
        PUBLIC
        Kokkos::kokkos
        specfem::assembly
        # Trilinos::all_selected_libs aggregates exactly the enabled packages
        # (Tpetra/Belos/Ifpack2/MueLu/Amesos2) from find_package(Trilinos).
        $<$<BOOL:${SPECFEM_ENABLE_TRILINOS}>:Trilinos::all_selected_libs>
)

target_compile_definitions(
        specfem_linear_system
        PUBLIC
        $<$<NOT:$<BOOL:${SPECFEM_ENABLE_TRILINOS}>>:-DNO_TRILINOS>
)

set_target_properties(specfem_linear_system PROPERTIES
    UNITY_BUILD $<NOT:$<BOOL:${__APPLE__}>>
)
