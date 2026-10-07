
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
            "  Load a TROMP module (e.g. `module load trilinos/17.1.1-cpu-native-nompi`)\n"
            "  or pass -DTrilinos_ROOT=<install-prefix> to cmake.")
    endif()

    # cmake/kokkos.cmake runs before this file and only reuses an existing Kokkos
    # when Kokkos_ROOT is set; otherwise it FetchContent's its own copy. That
    # would silently give SPECFEM++ a different Kokkos than the one Trilinos was
    # built against, which fails to compile or link (see docs/sections/getting_started/trilinos.rst).
    if(NOT Kokkos_ROOT AND NOT DEFINED ENV{Kokkos_ROOT})
        message(FATAL_ERROR
            "Kokkos_ROOT must be set when SPECFEM_ENABLE_TRILINOS is ON.\n"
            "  SPECFEM++ must reuse the SAME Kokkos that Trilinos was built against.\n"
            "  Point Kokkos_ROOT at the Kokkos install from your Trilinos build directory\n"
            "  (e.g. `module load trilinos/16.1.0-cuda-ampere80-mpi`, which exports both\n"
            "  Trilinos_ROOT and Kokkos_ROOT), or pass -DKokkos_ROOT=<path> to cmake.")
    endif()

    find_package(Trilinos REQUIRED
        COMPONENTS Tpetra Belos Ifpack2 MueLu Amesos2
        HINTS
            ${Trilinos_ROOT}
            ENV Trilinos_ROOT)

    add_compile_definitions(SPECFEM_ENABLE_TRILINOS)

    # Trilinos dylibs carry "@rpath/..." install names. On macOS, dyld ignores
    # LD_LIBRARY_PATH (all the TROMP modulefiles set) and SIP strips DYLD_*
    # under make/cmake; unlike ELF platforms, CMake also does not add imported
    # libraries' directories to the build rpath automatically. Embed the
    # Trilinos library directory so binaries run without environment setup.
    # Trilinos_DIR is <prefix>/<libdir>/cmake/Trilinos, so <libdir> is two
    # levels up.
    if(APPLE)
        get_filename_component(SPECFEM_TRILINOS_LIBRARY_DIR
            "${Trilinos_DIR}/../.." ABSOLUTE)
        list(APPEND CMAKE_BUILD_RPATH "${SPECFEM_TRILINOS_LIBRARY_DIR}")
        list(APPEND CMAKE_INSTALL_RPATH "${SPECFEM_TRILINOS_LIBRARY_DIR}")
    endif()

    message(STATUS "Found Trilinos ${Trilinos_VERSION}")
    message(STATUS "    DIR : ${Trilinos_DIR}")
    message(STATUS "    PKGS: ${Trilinos_PACKAGE_LIST}")

    list(POP_BACK CMAKE_MESSAGE_INDENT)

else()
    message(STATUS "Trilinos support is disabled. Set SPECFEM_ENABLE_TRILINOS to ON to enable it.")
endif()
