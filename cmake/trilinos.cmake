
if(SPECFEM_ENABLE_TRILINOS)

    message(STATUS "Trilinos support is enabled. Proceeding with Trilinos configuration.")

    list(APPEND CMAKE_MESSAGE_INDENT "  Trilinos: ")

    # Trilinos must be built against the SAME Kokkos as SPECFEM++. The TROMP
    # `trilinos/<variant>` modulefiles export Trilinos_ROOT and Kokkos_ROOT pointing
    # at one install; loading the module makes cmake/kokkos.cmake find_package that
    # Kokkos instead of FetchContent-ing a second copy. No FetchContent here -- the
    # install must be provided (cf. cmake/metis.cmake).
    if(NOT Trilinos_ROOT AND NOT DEFINED ENV{Trilinos_ROOT})
        message(FATAL_ERROR
            "Trilinos_ROOT must be set when SPECFEM_ENABLE_TRILINOS is ON.\n"
            "  Load a TROMP module (e.g. `module load trilinos/16.1.0-cuda-ampere80-mpi`)\n"
            "  or pass -DTrilinos_ROOT=<install-prefix> to cmake.")
    endif()

    find_package(Trilinos REQUIRED
        COMPONENTS Tpetra Belos Ifpack2 MueLu Amesos2
        HINTS
            ${Trilinos_ROOT}
            ENV Trilinos_ROOT)

    add_compile_definitions(SPECFEM_ENABLE_TRILINOS)

    message(STATUS "Found Trilinos ${Trilinos_VERSION}")
    message(STATUS "    DIR : ${Trilinos_DIR}")
    message(STATUS "    PKGS: ${Trilinos_PACKAGE_LIST}")

    list(POP_BACK CMAKE_MESSAGE_INDENT)

else()
    message(STATUS "Trilinos support is disabled. Set SPECFEM_ENABLE_TRILINOS to ON to enable it.")
endif()
