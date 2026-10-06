# Standalone C++ boundary around the Fortran globe model catalog. Assembly
# consumes this setup-time evaluator, so globe is configured before assembly.
add_library(specfem_globe STATIC model_evaluator.cpp)
add_library(specfem::globe ALIAS specfem_globe)

target_include_directories(specfem_globe
    PUBLIC
        ${CMAKE_SOURCE_DIR}/core
)

target_link_libraries(specfem_globe
    PUBLIC
        specfem::mpi
        specfem::utilities
)

target_link_libraries(specfem_globe
    PUBLIC
        globe_evaluator::globe_evaluator
)
