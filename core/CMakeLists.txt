# Add the specfem subdirectory which contains libraries
add_subdirectory(specfem)

# Create the unified specfem executable
add_executable(specfem specfem.cpp)

# Link the executable with our unified libraries
target_include_directories(specfem PRIVATE ${CMAKE_SOURCE_DIR}/core)
target_link_libraries(specfem
    PRIVATE
        specfem_unified
        Boost::program_options
        yaml-cpp::yaml-cpp
        # Add periodic tasks dependency when available
        $<TARGET_NAME_IF_EXISTS:periodic_tasks>
)

# Set executable properties
set_target_properties(specfem PROPERTIES
    OUTPUT_NAME "specfem"
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)
