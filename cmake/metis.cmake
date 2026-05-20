
if(SPECFEM_ENABLE_METIS)

    message(STATUS "METIS support is enabled. Proceeding with METIS configuration.")

    list(APPEND CMAKE_MESSAGE_INDENT "  METIS: ")

    if(NOT METIS_ROOT AND NOT DEFINED ENV{METIS_ROOT})
        message(FATAL_ERROR
            "METIS_ROOT must be set when SPECFEM_ENABLE_METIS is ON.\n"
            "  Pass -DMETIS_ROOT=<path> to cmake or export METIS_ROOT=<path>.")
    endif()

    find_path(METIS_INCLUDE_DIR
        NAMES metis.h
        PATH_SUFFIXES metis include/metis include
        HINTS
            ${METIS_ROOT}
            ENV METIS_ROOT
    )

    find_library(METIS_LIBRARY
        NAMES metis
        PATH_SUFFIXES lib lib64
        HINTS
            ${METIS_ROOT}
            ENV METIS_ROOT
    )

    if(NOT METIS_INCLUDE_DIR OR NOT METIS_LIBRARY)
        message(FATAL_ERROR
            "METIS not found under METIS_ROOT=${METIS_ROOT}.\n"
            "  Ensure METIS_ROOT points to a valid METIS installation prefix.")
    endif()

    message(STATUS "Using system-wide METIS")
    message(STATUS "    LIB: ${METIS_LIBRARY}")
    message(STATUS "    INC: ${METIS_INCLUDE_DIR}")

    mark_as_advanced(METIS_INCLUDE_DIR METIS_LIBRARY)

    if(NOT TARGET METIS::metis)
        add_library(METIS::metis UNKNOWN IMPORTED)
        set_target_properties(METIS::metis PROPERTIES
            IMPORTED_LOCATION             "${METIS_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${METIS_INCLUDE_DIR}"
        )
    endif()

    list(POP_BACK CMAKE_MESSAGE_INDENT)

else()
    message(STATUS "METIS support is disabled. Set SPECFEM_ENABLE_METIS to ON to enable it.")
endif()
