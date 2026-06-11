file(GLOB_RECURSE ASSEMBLY_SOURCE_FILES "*.cpp")

add_library(assembly ${ASSEMBLY_SOURCE_FILES})
add_library(specfem::assembly ALIAS assembly)

target_link_libraries(
        assembly
        PUBLIC
        specfem::macros
        $<$<BOOL:${SPECFEM_ENABLE_VTK}>:${VTK_LIBRARIES}>
        PRIVATE
        specfem::algorithms
        specfem::attenuation
        specfem::element
        specfem::element_coupling
        specfem::quadrature
        specfem::mesh
        specfem::mpi
        specfem::program_abort
        source_class
        jacobian
        point
        specfem::receivers
        shape_functions
        Kokkos::kokkos
        specfem::enums
        ${BOOST_LIBS}
        $<$<BOOL:${SPECFEM_ENABLE_VTK}>:${VTK_LIBRARIES}>
)

set_target_properties(assembly PROPERTIES
    UNITY_BUILD $<NOT:$<BOOL:${__INTEL_LLVM__}>>
    UNITY_BUILD_BATCH_SIZE $<IF:$<BOOL:${__INTEL_LLVM__}>,0,8>
)

target_compile_definitions(
    assembly
    PUBLIC
    $<$<NOT:$<BOOL:${SPECFEM_ENABLE_VTK}>>:-DNO_VTK>
)

target_include_directories(
        assembly
        INTERFACE
        ${HDF5_INCLUDE_DIRS}
)
