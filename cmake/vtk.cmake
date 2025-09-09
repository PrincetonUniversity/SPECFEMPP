if (SPECFEM_ENABLE_VTK)

    message(STATUS "VTK support is enabled. Proceeding with VTK configuration.")

    # Prepend the CMAKE_MESSAGE_INDENT variable to ensure proper indentation in messages
    list(APPEND CMAKE_MESSAGE_INDENT "  VTK: ")

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
    else()
        message(STATUS "VTK not found. Building without VTK.")
        set(SPECFEM_ENABLE_VTK OFF)
    endif()

    # Pop the indentation for VTK messages
    list(POP_BACK CMAKE_MESSAGE_INDENT)
else()
    set(VTK_FOUND OFF)
    set(SPECFEM_ENABLE_VTK OFF)
    message(STATUS "Building without VTK.")
endif()
