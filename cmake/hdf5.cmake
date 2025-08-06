

if (SPECFEM_ENABLE_HDF5)

    message(STATUS "HDF5 support is enabled. Proceeding with HDF5 configuration.")

    # Prepend the CMAKE_MESSAGE_INDENT variable to ensure proper indentation in messages
    list(APPEND CMAKE_MESSAGE_INDENT "  HDF5: ")

    # Install HDF5 as a dependency if not found
    find_package(HDF5 COMPONENTS CXX)

    if (NOT ${HDF5_FOUND})
        message(STATUS "HDF5 not found. Building without HDF5.")
        set(SPECFEM_ENABLE_HDF5 OFF CACHE BOOL "Disable HDF5 support" FORCE)
    else()
        message(STATUS "HDF5 libs/ and incs/:.")
        message(STATUS "    LIB:   ${HDF5_LIBRARIES}")
        message(STATUS "    INC:   ${HDF5_INCLUDE_DIRS}")
        message(STATUS "    LIBSO: ${HDF5_CXX_LIBRARIES}")
    endif()

    # Pop the indentation for HDF5 messages
    list(POP_BACK CMAKE_MESSAGE_INDENT)
else()
    message(STATUS "HDF5 support is disabled. Set SPECFEM_ENABLE_HDF5 to ON to enable it.")
    set(SPECFEM_ENABLE_HDF5 OFF CACHE BOOL "Disable HDF5 support" FORCE)
endif()
