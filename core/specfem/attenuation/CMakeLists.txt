file(GLOB_RECURSE SPECFEM_ATTENUATION_SOURCE_FILES "./*.cpp")

add_library(specfem_attenuation ${SPECFEM_ATTENUATION_SOURCE_FILES})
add_library(specfem::attenuation ALIAS specfem_attenuation)

set_target_properties(specfem_attenuation PROPERTIES
        UNITY_BUILD $<NOT:$<BOOL:${__APPLE__}>>
        UNITY_BUILD_BATCH_SIZE $<IF:$<BOOL:${__APPLE__}>,0,4>
)

target_link_libraries(
        specfem_attenuation
        PRIVATE
        Kokkos::kokkos
)
