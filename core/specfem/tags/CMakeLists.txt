 # Create header-only interface library for specfem::tags
add_library(specfem_tags INTERFACE)

# Create alias for consistent naming
add_library(specfem::tags ALIAS specfem_tags)

# Add include directories so the library can find its headers
target_include_directories(specfem_tags
    INTERFACE
        ${CMAKE_SOURCE_DIR}/core
        ${CMAKE_SOURCE_DIR}/include
)

# Link dependencies - tags library depends on enumerations
target_link_libraries(specfem_tags
    INTERFACE
        ${BOOST_LIBS}  # For boost preprocessor used in tag generation
        Kokkos::kokkos
)

# Set compile features for C++17 support (required for tag system)
target_compile_features(specfem_tags
    INTERFACE
        cxx_std_17
)
