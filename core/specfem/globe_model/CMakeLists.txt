# C++ binding for the SPECFEM3D_GLOBE model evaluator (issue #2001).
#
# This target always exists so higher-level model-dispatch code can depend on a
# single interface. Globe-enabled builds link the Fortran catalog; non-globe
# builds compile a throwing stub and do not pull in Fortran runtime libraries.
if(SPECFEM_BUILD_MESHFEM3D_GLOBE)
    add_library(specfem_globe_model STATIC evaluator.cpp)
else()
    add_library(specfem_globe_model STATIC stub_evaluator.cpp)
endif()
add_library(specfem::globe_model ALIAS specfem_globe_model)

target_include_directories(specfem_globe_model
    PUBLIC
        ${CMAKE_SOURCE_DIR}/core
)

target_link_libraries(specfem_globe_model
    PUBLIC
        specfem::mpi
)

if(SPECFEM_BUILD_MESHFEM3D_GLOBE)
    target_link_libraries(specfem_globe_model
        PUBLIC
            globe_evaluator::globe_evaluator
    )
endif()
