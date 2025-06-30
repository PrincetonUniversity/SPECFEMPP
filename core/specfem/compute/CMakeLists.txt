file(GLOB COMPUTE_SOURCE_FILES "./*.cpp" "./*/*.cpp")

add_library(compute ${COMPUTE_SOURCE_FILES})

target_link_libraries(
        compute
        enumerations
        quadrature
        mesh
        source_class
        jacobian
        point
        edge
        receiver_class
        Kokkos::kokkos
)


if (NOT VTK_CXX_BUILD)
        target_compile_definitions(
                compute
                PUBLIC -DNO_VTK
        )
else ()
        target_link_libraries(
                compute
                ${VTK_LIBRARIES}
                )

        # Only define the __APPLE__ compile definition if it is defined
        if (__APPLE__)
        target_compile_definitions(compute PRIVATE __APPLE__)
        endif(__APPLE__)
endif()

target_include_directories(
        compute
        INTERFACE
        ${HDF5_INCLUDE_DIRS}
)
