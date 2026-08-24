# ==============================================================================
# Test registration for SPECFEM++
#
# Invariants that the rest of the test tree relies on:
#
#   * SPECFEM_TEST_OUTPUT_DIR is absolute and is computed exactly once, by
#     specfem_init_tests(). It is a CACHE INTERNAL variable so that no
#     subdirectory can re-derive it from the (possibly relative) SPECFEMPP_TEST_DIR.
#     Deriving it per-scope is what used to write test output into the source tree:
#     file(WRITE) and install(DESTINATION) resolve relative paths against the
#     current source dir / install prefix, not the build tree.
#
#   * SPECFEM_TEST_OUTPUT_DIR is never a CMake binary directory. We write our own
#     CTestTestfile.cmake there, and CMake generates one in every binary directory;
#     pointing them at the same place would make the two clobber each other and
#     break `ctest --test-dir <build>/tests/unit-tests` (used by GitHub CI and
#     scripts/coverage.smk).
#
#   * Every test's WORKING_DIRECTORY is that one absolute directory, so a test
#     resolves its data the same way no matter where ctest was invoked from.
#
#   * Test binaries stay in the build tree. gtest_discover_tests() already writes
#     discovery files holding absolute paths to them, so nothing needs copying.
# ==============================================================================

# Initialize the test tree. Call once, from tests/CMakeLists.txt, before adding
# any test suite. This is the only place SPECFEMPP_TEST_DIR is read.
macro(specfem_init_tests)
    if(DEFINED SPECFEMPP_TEST_DIR)
        # Relative values resolve against the source tree; absolute ones pass through.
        get_filename_component(_specfem_test_out "${SPECFEMPP_TEST_DIR}"
            ABSOLUTE BASE_DIR "${CMAKE_SOURCE_DIR}")
        set(SPECFEM_TEST_DATA_MODE INSTALL CACHE INTERNAL "test data placement mode")
    else()
        # Deliberately not a CMake binary directory -- see the invariants above.
        set(_specfem_test_out "${CMAKE_BINARY_DIR}/tests/run")
        set(SPECFEM_TEST_DATA_MODE SYMLINK CACHE INTERNAL "test data placement mode")
        # Only clean the default location; never rm -rf a user-supplied directory.
        set_directory_properties(PROPERTIES ADDITIONAL_CLEAN_FILES "${_specfem_test_out}")
    endif()

    set(SPECFEM_TEST_OUTPUT_DIR "${_specfem_test_out}"
        CACHE INTERNAL "absolute working directory for all SPECFEM++ tests")
    file(MAKE_DIRECTORY "${SPECFEM_TEST_OUTPUT_DIR}")

    set_property(GLOBAL PROPERTY SPECFEM_TEST_TARGETS "")

    message(STATUS "SPECFEM++ tests run in ${SPECFEM_TEST_OUTPUT_DIR} "
                   "(${SPECFEM_TEST_DATA_MODE} mode)")
endmacro()

# Define a GoogleTest executable and register it with CTest.
#
#   specfem_add_test(<name>
#     SOURCES     <file>...            # required
#     LIBRARIES   <lib|genex|flag>...  # linked PRIVATE (on an executable, == PUBLIC)
#     DEFINITIONS <def>...             # target_compile_definitions, PRIVATE
#     INCLUDES    <dir>...             # target_include_directories, PRIVATE
#     PROPERTIES  <key> <value>...     # raw set_target_properties escape hatch
#     LABELS      <label>...           # ctest -L
#     MPI_RANKS   <n>                  # run under `mpirun -n <n>`
#     TIMEOUT     <seconds>            # per-test TIMEOUT property
#     NO_UNITY                         # UNITY_BUILD OFF for this target
#     NO_CTEST                         # build it, but do not register it with CTest
#   )
#
# Definition and registration happen together, so a target cannot be built but
# left unregistered -- which is how two serial tests and eleven MPI tests silently
# stopped running under CTest when the registration lists were maintained by hand.
# NO_CTEST is the explicit way to say "known not runnable yet"; it is deliberately
# noisy in the call site rather than an omission somewhere else in the file.
function(specfem_add_test name)
    set(_options NO_UNITY NO_CTEST)
    set(_one_value MPI_RANKS TIMEOUT)
    set(_multi_value SOURCES LIBRARIES DEFINITIONS INCLUDES PROPERTIES LABELS)
    cmake_parse_arguments(PARSE_ARGV 1 T "${_options}" "${_one_value}" "${_multi_value}")

    if(T_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "specfem_add_test(${name}): unrecognized arguments: ${T_UNPARSED_ARGUMENTS}")
    endif()
    if(NOT T_SOURCES)
        message(FATAL_ERROR "specfem_add_test(${name}): SOURCES is required")
    endif()
    if(NOT SPECFEM_TEST_OUTPUT_DIR)
        message(FATAL_ERROR "specfem_add_test(${name}): specfem_init_tests() has not run")
    endif()

    add_executable(${name} ${T_SOURCES})
    target_link_libraries(${name} PRIVATE ${T_LIBRARIES})

    if(T_DEFINITIONS)
        target_compile_definitions(${name} PRIVATE ${T_DEFINITIONS})
    endif()
    if(T_INCLUDES)
        target_include_directories(${name} PRIVATE ${T_INCLUDES})
    endif()
    if(T_NO_UNITY)
        set_target_properties(${name} PROPERTIES UNITY_BUILD OFF)
    endif()
    if(T_PROPERTIES)
        set_target_properties(${name} PROPERTIES ${T_PROPERTIES})
    endif()

    if(T_NO_CTEST)
        return()
    endif()

    # With an external test directory, that directory has to stand on its own: CI runs
    # ctest from it on a compute node that cannot see the build tree (the Jenkins
    # workspace lives on node-local /scratch, the test directory on shared GPFS). So
    # build the binary straight into it -- then the discovery file that
    # gtest_discover_tests() generates already names the binary at its final path, and
    # no path rewriting is needed. In the default in-tree mode the binaries stay in the
    # build tree, where scripts/coverage.smk expects to find them.
    if(SPECFEM_TEST_DATA_MODE STREQUAL "INSTALL")
        set_target_properties(${name} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${SPECFEM_TEST_OUTPUT_DIR}")
    endif()

    # Launcher. MPI_RANKS runs the test under `mpirun -n <ranks>`; in an MPI build
    # even the serial tests run under `mpirun -n 1` so that MPI_Init() succeeds.
    # This must precede gtest_discover_tests(): the GoogleTest module reads
    # CROSSCOMPILING_EMULATOR off the target at call time.
    if(T_MPI_RANKS)
        set(_ranks ${T_MPI_RANKS})
    elseif(SPECFEM_ENABLE_MPI)
        set(_ranks 1)
    else()
        set(_ranks "")
    endif()
    if(_ranks)
        set_target_properties(${name} PROPERTIES
            CROSSCOMPILING_EMULATOR "${MPIEXEC_EXECUTABLE};${MPIEXEC_NUMPROC_FLAG};${_ranks}")
    endif()

    # CTest properties. Multi-rank tests reserve their ranks and do not overlap
    # with each other. Serial tests keep CTest's default timeout.
    set(_properties "")
    if(T_MPI_RANKS)
        list(APPEND _properties PROCESSORS ${T_MPI_RANKS} RUN_SERIAL ON)
        if(NOT T_TIMEOUT)
            set(T_TIMEOUT 300)
        endif()
    elseif(SPECFEM_ENABLE_MPI)
        list(APPEND _properties PROCESSORS 1)
    endif()
    if(T_TIMEOUT)
        list(APPEND _properties TIMEOUT ${T_TIMEOUT})
    endif()
    if(T_LABELS)
        list(APPEND _properties LABELS "${T_LABELS}")
    endif()

    set(_discover_args
        DISCOVERY_MODE POST_BUILD
        DISCOVERY_TIMEOUT 300
        WORKING_DIRECTORY "${SPECFEM_TEST_OUTPUT_DIR}")
    if(_properties)
        list(APPEND _discover_args PROPERTIES ${_properties})
    endif()
    gtest_discover_tests(${name} ${_discover_args})

    # Obtain the discovery filename from the include file registered by CMake instead
    # of reconstructing it. CMake 3.x uses <name>[1]_tests.cmake, whereas CMake 4.4+
    # uses a hash in the filename. Copy it into the test directory so the
    # CTestTestfile.cmake there can include it without reaching back into the build
    # tree. Added after gtest_discover_tests() so it runs after the POST_BUILD step
    # that generates the file.
    get_property(_test_include_files DIRECTORY PROPERTY TEST_INCLUDE_FILES)
    list(GET _test_include_files -1 _test_include_file)
    string(REGEX REPLACE "_include\\.cmake$" "_tests.cmake"
        _test_discovery_file "${_test_include_file}")
    if(_test_discovery_file STREQUAL _test_include_file)
        message(FATAL_ERROR
            "specfem_add_test(${name}): unexpected GoogleTest include filename: "
            "${_test_include_file}")
    endif()
    add_custom_command(TARGET ${name} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${_test_discovery_file}"
            "${SPECFEM_TEST_OUTPUT_DIR}/${name}_tests.cmake"
        COMMENT "Copying ${name} test discovery file to ${SPECFEM_TEST_OUTPUT_DIR}")

    set_property(GLOBAL APPEND PROPERTY SPECFEM_TEST_TARGETS ${name})
endfunction()

# Make test data directories (given relative to the calling suite) available under
# SPECFEM_TEST_OUTPUT_DIR: symlinked for an in-tree run, installed for an external one.
#
# These directories hold the test sources as well as the data, so the install copies
# only what a test reads at runtime -- C++ sources and build files are excluded rather
# than duplicated into the test directory.
function(specfem_add_test_data)
    foreach(dir_name IN LISTS ARGN)
        if(SPECFEM_TEST_DATA_MODE STREQUAL "SYMLINK")
            file(CREATE_LINK
                "${CMAKE_CURRENT_SOURCE_DIR}/${dir_name}"
                "${SPECFEM_TEST_OUTPUT_DIR}/${dir_name}"
                SYMBOLIC)
        else()
            # DESTINATION is absolute, so the install prefix is deliberately ignored.
            install(DIRECTORY ${dir_name}
                DESTINATION "${SPECFEM_TEST_OUTPUT_DIR}"
                COMPONENT tests
                PATTERN "*.cpp" EXCLUDE
                PATTERN "*.hpp" EXCLUDE
                PATTERN "*.tpp" EXCLUDE
                PATTERN "CMakeLists.txt" EXCLUDE)
        endif()
    endforeach()
endfunction()

# Finalize the test tree. Call once, from tests/CMakeLists.txt, after every suite
# has been added. The CTestTestfile and the install target are per-output-directory,
# not per-suite, so emitting them inside a suite would redefine them as soon as a
# second suite appeared.
function(specfem_finalize_tests)
    get_property(_targets GLOBAL PROPERTY SPECFEM_TEST_TARGETS)
    if(NOT _targets)
        message(FATAL_ERROR "specfem_finalize_tests(): no tests were registered")
    endif()

    # One CTestTestfile.cmake covering every suite, so `ctest --test-dir
    # <SPECFEM_TEST_OUTPUT_DIR>` (equivalently, cd + ctest) works. It must reference
    # ONLY files inside SPECFEM_TEST_OUTPUT_DIR: CI runs ctest on a compute node that
    # cannot see the build tree, so any build-tree path here is a dangling include.
    # Each <name>_tests.cmake is copied in by the POST_BUILD step in specfem_add_test();
    # a target that failed to build simply has none, hence the NOT_BUILT fallback.
    set(_ctest_content "# Generated by specfem_finalize_tests(). Do not edit.\n")
    foreach(test_target IN LISTS _targets)
        string(APPEND _ctest_content
            "if(EXISTS \"${SPECFEM_TEST_OUTPUT_DIR}/${test_target}_tests.cmake\")\n"
            "  include(\"${SPECFEM_TEST_OUTPUT_DIR}/${test_target}_tests.cmake\")\n"
            "else()\n"
            "  add_test(${test_target}_NOT_BUILT ${test_target}_NOT_BUILT)\n"
            "endif()\n")
    endforeach()
    file(WRITE "${SPECFEM_TEST_OUTPUT_DIR}/CTestTestfile.cmake" "${_ctest_content}")

    if(SPECFEM_TEST_DATA_MODE STREQUAL "INSTALL")
        # Installing from <build>/tests recurses into every suite below it.
        add_custom_target(install_test_data ALL
            COMMAND ${CMAKE_COMMAND} -E make_directory "${SPECFEM_TEST_OUTPUT_DIR}"
            COMMAND ${CMAKE_COMMAND} --install . --component tests
            WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
            COMMENT "Installing test data to ${SPECFEM_TEST_OUTPUT_DIR}")

        # DISCOVERY_MODE POST_BUILD runs each test binary from SPECFEM_TEST_OUTPUT_DIR
        # at link time, and some binaries read their data at static-init (that is,
        # even under --gtest_list_tests). So the data has to be in place first.
        foreach(test_target IN LISTS _targets)
            add_dependencies(${test_target} install_test_data)
        endforeach()
    endif()
endfunction()
