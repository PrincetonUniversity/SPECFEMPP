if (SPECFEM_ENABLE_VTK)

    find_package(VTK COMPONENTS
        CommonColor
        CommonCore
        FiltersSources
        InteractionStyle
        RenderingContextOpenGL2
        RenderingCore
        RenderingFreeType
        RenderingGL2PSOpenGL2
        RenderingOpenGL2
    )

    if (VTK_FOUND)
        message(STATUS "VTK libs/ and incs/:")
        message(STATUS "    LIB:   ${VTK_LIBRARY_DIRS}")
        message(STATUS "    INC:   ${VTK_INCLUDE_DIRS}")
        message(STATUS "    LIBSO: ${VTK_LIBRARIES}")
    else(NOT VTK_FOUND)
        message(STATUS "VTK not found. Building without VTK.")
        set(VTK_CXX_BUILD OFF)
    endif()
else ()
    set(VTK_FOUND OFF)
    set(SPECFEM_ENABLE_VTK OFF)
    message(STATUS "Building without VTK.")
endif()
