# C++ binding for the SPECFEM3D_GLOBE model evaluator (issue #2001).
#
# Deliberately NOT part of specfem::core, and not linked into the specfem
# executable. This target is the only thing in core/ that pulls in a Fortran
# archive and the Fortran runtime, so it stays opt-in until the evaluator is
# actually consumed by the assembly property build. Keeping it off the
# production link line also keeps the coverage preset unaffected -- see the
# note at CMakeLists.txt:269-273 about gfortran and -fprofile-instr-generate.
add_library(specfem_globe_model STATIC evaluator.cpp)
add_library(specfem::globe_model ALIAS specfem_globe_model)

target_include_directories(specfem_globe_model
    PUBLIC
        ${CMAKE_SOURCE_DIR}/core
)

target_link_libraries(specfem_globe_model
    PUBLIC
        globe_evaluator::globe_evaluator
        specfem::mpi
)
