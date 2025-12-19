# Logger options library
# Program library - contains context and execution logic
file(GLOB_RECURSE PROGRAM_SOURCES "*.cpp")
add_library(specfem_logger STATIC ${PROGRAM_SOURCES})
add_library(specfem::logger ALIAS specfem_logger)

target_include_directories(specfem_logger
    PUBLIC
        ${CMAKE_SOURCE_DIR}/core
)

target_link_libraries(specfem_logger
    PUBLIC
        ${BOOST_LIBS}
        specfem::mpi
)
