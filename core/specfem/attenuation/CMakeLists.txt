# Create header-only interface library for specfem::attenuation
add_library(specfem_attenuation INTERFACE)

# Create alias for consistent naming
add_library(specfem::attenuation ALIAS specfem_attenuation)

# Add include directories so the library can find its headers
target_include_directories(specfem_attenuation
    INTERFACE
        ${CMAKE_SOURCE_DIR}/core
)

target_link_libraries(specfem_attenuation INTERFACE
        Kokkos::kokkos
)
