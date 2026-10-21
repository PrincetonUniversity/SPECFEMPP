# Injection plugin loader: verifies, loads, and wraps wavefield-injection
# plugins built with the plugin SDK (sdk/). Deliberately free of Kokkos, MPI,
# and the logger so it can be tested in isolation.
file(GLOB_RECURSE INJECTION_SOURCES "*.cpp")
add_library(specfem_injection STATIC ${INJECTION_SOURCES})
add_library(specfem::injection ALIAS specfem_injection)

target_include_directories(specfem_injection PUBLIC ${CMAKE_SOURCE_DIR}/core)

target_link_libraries(specfem_injection
    PUBLIC
        specfem::plugin_sdk
    PRIVATE
        ${CMAKE_DL_LIBS}
)
