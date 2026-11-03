# Injection plugin loader: verifies, loads, and wraps wavefield-injection
# plugins built with the plugin SDK (sdk/). Deliberately free of Kokkos, MPI,
# and the logger, mirroring the plugin ABI boundary; the Kokkos/MPI side lives
# in session/.
file(GLOB INJECTION_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
add_library(specfem_injection STATIC ${INJECTION_SOURCES})
add_library(specfem::injection ALIAS specfem_injection)

target_include_directories(specfem_injection PUBLIC ${CMAKE_SOURCE_DIR}/core)

target_link_libraries(specfem_injection
    PUBLIC
        specfem::plugin_sdk
    PRIVATE
        ${CMAKE_DL_LIBS}
)

add_subdirectory(session)
