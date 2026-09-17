# Standalone C++ boundary around the optional Fortran globe model catalog.
# It remains separate from specfem::io because that general target depends on
# assembly, while assembly consumes this setup-time oracle.
if(SPECFEM_BUILD_MESHFEM3D_GLOBE)
    add_library(specfem_io_globe_model STATIC globe_model.cpp)
else()
    add_library(specfem_io_globe_model STATIC stub.cpp)
endif()
add_library(specfem::io_globe_model ALIAS specfem_io_globe_model)

target_include_directories(specfem_io_globe_model
    PUBLIC
        ${CMAKE_SOURCE_DIR}/core
)

target_link_libraries(specfem_io_globe_model
    PUBLIC
        specfem::mpi
        specfem::utilities
)

if(SPECFEM_BUILD_MESHFEM3D_GLOBE)
    target_link_libraries(specfem_io_globe_model
        PUBLIC
            globe_evaluator::globe_evaluator
    )
endif()
