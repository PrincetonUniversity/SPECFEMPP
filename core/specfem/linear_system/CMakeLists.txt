file(GLOB_RECURSE LINEAR_SYSTEM_SOURCE_FILES "./*.cpp" "./**/*.cpp")

add_library(specfem_linear_system ${LINEAR_SYSTEM_SOURCE_FILES})
add_library(specfem::linear_system ALIAS specfem_linear_system)

target_link_libraries(
        specfem_linear_system
        PUBLIC
        Kokkos::kokkos
        specfem::assembly
        # Tpetra (matrices/vectors), Belos (GMRES), and Ifpack2 (RILUK /
        # relaxation preconditioners) are linked individually; all three are
        # templated on the Tpetra scalar and instantiate for float builds.
        # Trilinos::all_selected_libs stays unusable against the float-only
        # (Tpetra_INST_DOUBLE=OFF) TROMP Trilinos install -- MueLu references
        # Xpetra::Matrix<double, ...> symbols unconditionally, which that
        # install doesn't instantiate. MueLu/Amesos2 remain unlinked until the
        # install is revisited (issue #1984).
        $<$<BOOL:${SPECFEM_ENABLE_TRILINOS}>:Tpetra::all_libs>
        $<$<BOOL:${SPECFEM_ENABLE_TRILINOS}>:Belos::all_libs>
        $<$<BOOL:${SPECFEM_ENABLE_TRILINOS}>:Ifpack2::all_libs>
)

target_compile_definitions(
        specfem_linear_system
        PUBLIC
        $<$<NOT:$<BOOL:${SPECFEM_ENABLE_TRILINOS}>>:-DNO_TRILINOS>
)

set_target_properties(specfem_linear_system PROPERTIES
    UNITY_BUILD $<NOT:$<BOOL:${__APPLE__}>>
)
