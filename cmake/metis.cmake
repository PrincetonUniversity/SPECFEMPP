
if(SPECFEM_ENABLE_METIS)

    message(STATUS "METIS support is enabled. Proceeding with METIS configuration.")

    list(APPEND CMAKE_MESSAGE_INDENT "  METIS: ")

    if(NOT METIS_ROOT AND NOT DEFINED ENV{METIS_ROOT})
        message(FATAL_ERROR
            "METIS_ROOT must be set when SPECFEM_ENABLE_METIS is ON.\n"
            "  Pass -DMETIS_ROOT=<path> to cmake or export METIS_ROOT=<path>.")
    endif()

    # GKlib's packaging depends on the METIS version:
    #   - METIS 5.2.x (KarypisLab) ships GKlib as a separate cmake config package.
    #   - METIS 5.1.x (e.g. Homebrew) bundles GKlib inside libmetis; there is no
    #     standalone GKlib library or config to find.
    # So look for GKlib but do not require it; link it only when it is found.
    # The GKLIB_DIR env var is set by the gklib module file on HPC systems.
    if(DEFINED ENV{GKLIB_DIR})
        list(APPEND CMAKE_PREFIX_PATH $ENV{GKLIB_DIR})
    endif()
    find_package(GKlib QUIET)
    if(GKlib_FOUND)
        message(STATUS "Found standalone GKlib; linking METIS against GKlib::GKlib")
    else()
        message(STATUS "No standalone GKlib found; assuming GKlib is bundled in libmetis")
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
        if(GKlib_FOUND)
            set_target_properties(METIS::metis PROPERTIES
                INTERFACE_LINK_LIBRARIES "GKlib::GKlib")
        endif()
    endif()

    list(POP_BACK CMAKE_MESSAGE_INDENT)

else()
    message(STATUS "METIS support is disabled. Set SPECFEM_ENABLE_METIS to ON to enable it.")
endif()
