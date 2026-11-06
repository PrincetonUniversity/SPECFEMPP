
if(SPECFEM_ENABLE_TENSOROPS)

    message(STATUS "TensorOperations support is enabled.")

    list(APPEND CMAKE_MESSAGE_INDENT "  TensorOperations: ")

    # TensorOperations is header-only, but its own CMakeLists.txt unconditionally
    # FetchContent's a SECOND Kokkos (develop) plus GoogleTest and defines its test
    # targets with no PROJECT_IS_TOP_LEVEL guard -- so it is never consumed via
    # add_subdirectory or FetchContent. Instead an INTERFACE target is built over
    # its include/ directory and linked against SPECFEM++'s own Kokkos. The
    # checkout must be provided, like every optional dependency
    # (cf. cmake/trilinos.cmake); the library also has no license yet, which rules
    # out vendoring.
    if(NOT SPECFEM_TENSOROPS_ROOT AND DEFINED ENV{SPECFEM_TENSOROPS_ROOT})
        set(SPECFEM_TENSOROPS_ROOT "$ENV{SPECFEM_TENSOROPS_ROOT}")
    endif()
    if(NOT SPECFEM_TENSOROPS_ROOT)
        message(FATAL_ERROR
            "SPECFEM_TENSOROPS_ROOT must be set when SPECFEM_ENABLE_TENSOROPS is ON.\n"
            "  Pass -DSPECFEM_TENSOROPS_ROOT=<checkout> (the TensorOperations repository\n"
            "  root containing include/TensorOperations/) or export SPECFEM_TENSOROPS_ROOT.")
    endif()
    if(NOT EXISTS "${SPECFEM_TENSOROPS_ROOT}/include/TensorOperations/LevelGraph.hpp")
        message(FATAL_ERROR
            "SPECFEM_TENSOROPS_ROOT (${SPECFEM_TENSOROPS_ROOT}) does not look like a\n"
            "  TensorOperations checkout: include/TensorOperations/LevelGraph.hpp is missing.")
    endif()

    add_library(specfem_tensorops INTERFACE)
    add_library(specfem::tensorops ALIAS specfem_tensorops)
    target_include_directories(specfem_tensorops INTERFACE
        "${SPECFEM_TENSOROPS_ROOT}/include")
    target_link_libraries(specfem_tensorops INTERFACE Kokkos::kokkos)
    target_compile_features(specfem_tensorops INTERFACE cxx_std_20)
    # The library's compile-time contraction plans nest brackets past Clang's
    # default limit; mirrors the flag in its own CMakeLists.txt.
    target_compile_options(specfem_tensorops INTERFACE
        $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>:-fbracket-depth=2048>)

    add_compile_definitions(SPECFEM_ENABLE_TENSOROPS)

    message(STATUS "Found TensorOperations at ${SPECFEM_TENSOROPS_ROOT}")

    list(POP_BACK CMAKE_MESSAGE_INDENT)

else()
    message(STATUS "TensorOperations support is disabled. Set SPECFEM_ENABLE_TENSOROPS to ON to enable it.")
endif()
