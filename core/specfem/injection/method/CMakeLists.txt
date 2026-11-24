# Injection methods: how the solver applies an incident field. The solver sees
# only coupling_method.hpp (header-only, assembly types only); the program
# builds methods from the registry and hands them the provider session.
file(GLOB_RECURSE INJECTION_METHODS_SOURCES "*.cpp")
add_library(specfem_injection_methods STATIC ${INJECTION_METHODS_SOURCES})
add_library(specfem::injection_methods ALIAS specfem_injection_methods)

target_include_directories(specfem_injection_methods
    PUBLIC
        ${CMAKE_SOURCE_DIR}/core
)

target_link_libraries(specfem_injection_methods
    PUBLIC
        specfem::injection_points
)
