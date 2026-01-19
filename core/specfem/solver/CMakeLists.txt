file(GLOB_RECURSE SOLVER_SOURCE_FILES "./*.cpp" "./**/*.cpp")

add_library(
        specfem_solver
        ${SOLVER_SOURCE_FILES}
)
add_library(specfem::solver ALIAS specfem_solver)

target_link_libraries(
        specfem_solver
        Kokkos::kokkos
        specfem::timescheme
        specfem::kokkos_kernels
        specfem::periodic_tasks
        specfem::logger
)
