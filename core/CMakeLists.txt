# Add the specfem subdirectory which contains libraries
add_subdirectory(specfem)

# Create the unified specfem executable
add_executable(specfem specfem.cpp)

# Link the executable with our unified libraries
target_link_libraries(specfem
    PRIVATE
        specfem::core
        Boost::program_options
        yaml-cpp::yaml-cpp
)

# Set executable properties
set_target_properties(specfem PROPERTIES
    OUTPUT_NAME "specfem"
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)
