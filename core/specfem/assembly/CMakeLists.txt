file(GLOB_RECURSE COMPUTE_SOURCE_FILES "./*.cpp" "./**/*.cpp")

add_library(assembly ${COMPUTE_SOURCE_FILES})
add_library(specfem::assembly ALIAS assembly)

target_link_libraries(
        assembly
        algorithms
        enumerations
        quadrature
        mesh
        source_class
        jacobian
        point
        edge
        specfem::receivers
        shape_functions
        Kokkos::kokkos
        ${BOOST_LIBS}
)


if (NOT VTK_CXX_BUILD)
        target_compile_definitions(
                assembly
                PUBLIC -DNO_VTK
        )
else ()
        target_link_libraries(
                assembly
                ${VTK_LIBRARIES}
                )

        # Only define the __APPLE__ compile definition if it is defined
        if (__APPLE__)
        target_compile_definitions(assembly PRIVATE __APPLE__)
        endif(__APPLE__)
endif()

target_include_directories(
        assembly
        INTERFACE
        ${HDF5_INCLUDE_DIRS}
)
