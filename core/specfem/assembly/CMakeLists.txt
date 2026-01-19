file(GLOB_RECURSE COMPUTE_SOURCE_FILES "./*.cpp" "./**/*.cpp")

add_library(assembly ${COMPUTE_SOURCE_FILES})
add_library(specfem::assembly ALIAS assembly)

target_link_libraries(
        assembly
        PUBLIC
        specfem::macros
        $<$<BOOL:${SPECFEM_ENABLE_VTK}>:${VTK_LIBRARIES}>
        PRIVATE
        specfem::algorithms
        enumerations
        specfem::quadrature
        specfem::mesh
        source_class
        jacobian
        point
        specfem::receivers
        shape_functions
        Kokkos::kokkos
        enumerations
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
