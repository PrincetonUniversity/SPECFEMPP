file(GLOB_RECURSE RUNTIME_CONFIGURATION_SOURCE_FILES "./*.cpp" "./**/*.cpp")

add_library(specfem_runtime_configuration ${RUNTIME_CONFIGURATION_SOURCE_FILES})
add_library(specfem::runtime_configuration ALIAS specfem_runtime_configuration)

target_link_libraries(
        specfem_runtime_configuration
        specfem::quadrature
        specfem::timescheme
        specfem::receivers
        yaml-cpp
        specfem::periodic_tasks
        writer
        reader
        specfem::kokkos_kernels
        specfem::solver
        ${BOOST_LIBS}
)
