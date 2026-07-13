file(GLOB_RECURSE LINEAR_SYSTEM_SOURCE_FILES "./*.cpp" "./**/*.cpp")

add_library(specfem_linear_system ${LINEAR_SYSTEM_SOURCE_FILES})
add_library(specfem::linear_system ALIAS specfem_linear_system)

target_link_libraries(
        specfem_linear_system
        PUBLIC
        Kokkos::kokkos
        specfem::assembly
        # Only Tpetra is linked for now: trilinos_smoke_test() is the sole
        # consumer today. Trilinos::all_selected_libs (Belos/Ifpack2/MueLu/
        # Amesos2) fails to link against the float-only (Tpetra_INST_DOUBLE=OFF)
        # TROMP Trilinos install -- MueLu references Xpetra::Matrix<double, ...>
        # symbols unconditionally, which that install doesn't instantiate. Link
        # the other packages once real assembly/solver code needs them and the
        # install is revisited.
        $<$<BOOL:${SPECFEM_ENABLE_TRILINOS}>:Tpetra::all_libs>
)

target_compile_definitions(
        specfem_linear_system
        PUBLIC
        $<$<NOT:$<BOOL:${SPECFEM_ENABLE_TRILINOS}>>:-DNO_TRILINOS>
)

set_target_properties(specfem_linear_system PROPERTIES
    UNITY_BUILD $<NOT:$<BOOL:${__APPLE__}>>
)
