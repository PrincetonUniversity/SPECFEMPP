file(GLOB_RECURSE SOLVER_SOURCE_FILES "./*.cpp" "./**/*.cpp")

add_library(
        specfem_solver
        ${SOLVER_SOURCE_FILES}
)
add_library(specfem::solver ALIAS specfem_solver)

# turn of unity for the solver library
set_target_properties(specfem_solver PROPERTIES UNITY_BUILD OFF)

target_link_libraries(
        specfem_solver
        Kokkos::kokkos
        specfem::timescheme
        specfem::periodic_tasks
        specfem::logger
        specfem::io
        specfem::assembly
        specfem::compute
)
