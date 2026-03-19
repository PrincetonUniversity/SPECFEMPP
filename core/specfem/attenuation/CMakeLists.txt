file(GLOB_RECURSE ATTENUATION_SOURCES "*.cpp")
add_library(specfem_attenuation STATIC ${ATTENUATION_SOURCES})

# Create alias for consistent naming
add_library(specfem::attenuation ALIAS specfem_attenuation)

# Add include directories so the library can find its headers
target_include_directories(specfem_attenuation
    PUBLIC
        ${CMAKE_SOURCE_DIR}/core
)

target_link_libraries(specfem_attenuation PUBLIC
        Kokkos::kokkos
)
