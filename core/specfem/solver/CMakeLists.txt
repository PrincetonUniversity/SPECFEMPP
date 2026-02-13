file(GLOB_RECURSE SOLVER_SOURCE_FILES "./*.cpp" "./**/*.cpp")

add_library(
        specfem_solver
        ${SOLVER_SOURCE_FILES}
)
add_library(specfem::solver ALIAS specfem_solver)

target_precompile_headers(specfem_solver REUSE_FROM specfem_pch)

target_link_libraries(
        specfem_solver
        Kokkos::kokkos
        specfem::timescheme
        specfem::periodic_tasks
        specfem::logger
        specfem::io
        specfem::assembly
)
