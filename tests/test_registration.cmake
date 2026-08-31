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

# Launcher every multi-rank MPI test runs under, resolved once here at file scope so it
# does not depend on which listfile is being processed when specfem_add_test() is called.
# The rank count is appended per test. This file is included from tests/CMakeLists.txt,
# which is added after find_package(MPI), so the FindMPI results are available.
#
# It is used only by ctest, never during a build -- see the rank-count note in
# specfem_add_test() for why that matters when the launcher is `srun`.
#
# SPECFEM_MPI_TEST_COMMAND / _NUMPROC_FLAG (declared in the top-level CMakeLists.txt) are
# empty by default, meaning "use what FindMPI located" -- an absolute path to the loaded
# module's mpiexec, rather than a bare `mpirun` resolved off PATH at test time. Both are
# lists, so a launcher needing arguments works: -DSPECFEM_MPI_TEST_COMMAND="srun;--mpi=pmix".
if(SPECFEM_ENABLE_MPI)
    if(SPECFEM_MPI_TEST_COMMAND)
        set(_specfem_mpi_launch ${SPECFEM_MPI_TEST_COMMAND})
    else()
        set(_specfem_mpi_launch ${MPIEXEC_EXECUTABLE})
    endif()
    if(SPECFEM_MPI_TEST_NUMPROC_FLAG)
        list(APPEND _specfem_mpi_launch ${SPECFEM_MPI_TEST_NUMPROC_FLAG})
    else()
        list(APPEND _specfem_mpi_launch ${MPIEXEC_NUMPROC_FLAG})
    endif()
    set(SPECFEM_MPI_LAUNCH_COMMAND "${_specfem_mpi_launch}"
        CACHE INTERNAL "MPI launcher + numproc flag; the rank count is appended per test")
endif()

# Extract the GoogleTest cases declared in <source>... as "<display>|<filter>" pairs.
#
# Multi-rank tests are registered from these names instead of by gtest_discover_tests(),
# which cannot be used there: it runs the binary with --gtest_list_tests through the same
# executor it puts in front of the real test command (CMake merges TEST_LAUNCHER and
# CROSSCOMPILING_EMULATOR into one `test_executor`, used for both), so listing under
# `mpiexec -n <ranks>` makes every rank print to one merged stdout and registers each case
# once per rank. No target property can tell the two invocations apart.
#
# Every source is marked as a configure dependency, so adding or renaming a TEST re-runs
# CMake rather than silently leaving the case unregistered.
function(specfem_gtest_case_names out_var)
    set(_names "")
    foreach(_src IN LISTS ARGN)
        get_filename_component(_abs "${_src}" ABSOLUTE
            BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
        if(NOT EXISTS "${_abs}")
            continue()
        endif()
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_abs}")
        file(STRINGS "${_abs}" _lines
            REGEX "^TEST(_F|_P)?\\([A-Za-z0-9_]+,[ \t]*[A-Za-z0-9_]+\\)")
        foreach(_line IN LISTS _lines)
            string(REGEX MATCH "^TEST(_F|_P)?\\(([A-Za-z0-9_]+),[ \t]*([A-Za-z0-9_]+)\\)"
                _matched "${_line}")
            set(_case "${CMAKE_MATCH_2}.${CMAKE_MATCH_3}")
            if(CMAKE_MATCH_1 STREQUAL "_P")
                # A parametrized case is really <Prefix>/<Suite>.<Case>/<index>. Neither
                # the INSTANTIATE_TEST_SUITE_P prefix nor the number of values is visible
                # from this declaration, so one entry covers all of them.
                list(APPEND _names "${_case}|*${_case}/*")
            else()
                list(APPEND _names "${_case}|${_case}")
            endif()
        endforeach()
    endforeach()
    list(REMOVE_DUPLICATES _names)
    set(${out_var} "${_names}" PARENT_SCOPE)
endfunction()

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

    # Declaring MPI_RANKS is what makes a target part of the MPI test suite: it runs under
    # `<launcher> -n <ranks>`, at one rank as much as at four. A test that does not declare
    # it is an ordinary test that merely happens to be built in an MPI configuration, and
    # runs bare on singleton MPI_Init (supported by OpenMPI, MPICH and Intel MPI alike).
    #
    # That split is also what keeps the launcher out of the build. gtest_discover_tests()
    # executes each binary at link time (DISCOVERY_MODE POST_BUILD) through
    # CROSSCOMPILING_EMULATOR, and .jenkins/mpi_compiler_checks.gvy builds on a login node
    # outside any Slurm allocation -- putting `srun` in front of the ~50 ordinary test
    # binaries there would submit ~50 jobs during `cmake --build`. Every MPI-suite target
    # skips discovery entirely (add_test below), so the launcher is only ever invoked by
    # ctest, which that job runs inside `salloc`.
    if(T_MPI_RANKS)
        set(_ranks ${T_MPI_RANKS})
        if(NOT T_TIMEOUT)
            set(T_TIMEOUT 300)
        endif()
    else()
        set(_ranks 0)
    endif()

    # Every MPI-suite test carries the MPI label, which is how CI selects them
    # (.jenkins/mpi_compiler_checks.gvy runs `ctest -L MPI`). Names cannot be used for
    # that: the entries registered below are prefixed with the target, which is lower case.
    set(_labels "")
    if(T_MPI_RANKS)
        set(_labels MPI)
    endif()
    if(T_LABELS)
        list(APPEND _labels ${T_LABELS})
    endif()

    # MPI-suite tests are registered case by case from names grepped out of the sources,
    # because gtest_discover_tests() cannot list a binary that has to run under a launcher
    # -- see specfem_gtest_case_names() above. Ordinary tests keep discovery.
    if(T_MPI_RANKS)
        specfem_gtest_case_names(_cases ${T_SOURCES})
        if(NOT _cases)
            message(FATAL_ERROR
                "specfem_add_test(${name}): MPI_RANKS ${_ranks} is set but no "
                "TEST/TEST_F/TEST_P declaration was found in SOURCES. Either the sources "
                "moved or the extraction regex stopped matching them; registering nothing "
                "would make this target silently stop being tested.")
        endif()

        # FAIL_REGULAR_EXPRESSION guards the one way filter-based registration can pass
        # while running nothing: GoogleTest treats a filter matching no test as a warning
        # and still exits 0 (ShouldWarnIfNoTestsMatchFilter in gtest.cc), so a case renamed
        # between configures would otherwise stay green. There is deliberately no
        # SKIP_REGULAR_EXPRESSION -- that only ever existed to neutralize the "[  SKIPPED ]"
        # default gtest_discover_tests() sets, which under merged rank stdout let a
        # GTEST_SKIP() on an excluded rank mask a real failure on an active one.
        set(_mpi_properties
            WORKING_DIRECTORY "${SPECFEM_TEST_OUTPUT_DIR}"
            PROCESSORS ${_ranks}
            RUN_SERIAL ON
            TIMEOUT ${T_TIMEOUT}
            LABELS "${_labels}"
            FAIL_REGULAR_EXPRESSION "did not match any test")

        # The same registrations are also written into SPECFEM_TEST_OUTPUT_DIR so that
        # directory stands on its own (see specfem_finalize_tests). file(GENERATE) rather
        # than the POST_BUILD copy used below, because there is no discovery file to copy;
        # it expands $<TARGET_FILE:> so the generated commands name the binary absolutely.
        set(_standalone_props "")
        foreach(_prop IN LISTS _mpi_properties)
            string(APPEND _standalone_props " [==[${_prop}]==]")
        endforeach()
        set(_standalone_launch "")
        foreach(_arg IN LISTS SPECFEM_MPI_LAUNCH_COMMAND)
            string(APPEND _standalone_launch " [==[${_arg}]==]")
        endforeach()
        set(_standalone "# Generated by specfem_add_test(${name}). Do not edit.\n")

        foreach(_entry IN LISTS _cases)
            string(REGEX REPLACE "\\|.*$" "" _display "${_entry}")
            string(REGEX REPLACE "^[^|]*\\|" "" _filter "${_entry}")
            # Prefixed with the target because two targets can compile the same source,
            # and do: assembly_mpi_dim3_tests and assembly_mpi_dim3_8proc_tests share
            # three TEST_P declarations, which would collide as bare <Suite>.<Case>.
            set(_test ${name}.${_display})

            add_test(NAME ${_test}
                COMMAND ${SPECFEM_MPI_LAUNCH_COMMAND} ${_ranks}
                        $<TARGET_FILE:${name}> --gtest_filter=${_filter})
            set_tests_properties(${_test} PROPERTIES ${_mpi_properties})

            string(APPEND _standalone
                "add_test([==[${_test}]==]${_standalone_launch} [==[${_ranks}]==]"
                " [==[$<TARGET_FILE:${name}>]==] [==[--gtest_filter=${_filter}]==])\n"
                "set_tests_properties([==[${_test}]==] PROPERTIES${_standalone_props})\n")
        endforeach()

        file(GENERATE
            OUTPUT "${SPECFEM_TEST_OUTPUT_DIR}/${name}_tests.cmake"
            CONTENT "${_standalone}")

        set_property(GLOBAL APPEND PROPERTY SPECFEM_TEST_TARGETS ${name})
        return()
    endif()

    # Ordinary tests are discovered and run bare -- deliberately no
    # CROSSCOMPILING_EMULATOR, see the note on MPI_RANKS above.

    # CTest properties. These tests keep CTest's default timeout.
    set(_properties "")
    if(SPECFEM_ENABLE_MPI)
        list(APPEND _properties PROCESSORS 1)
    endif()
    if(T_TIMEOUT)
        list(APPEND _properties TIMEOUT ${T_TIMEOUT})
    endif()
    if(_labels)
        list(APPEND _properties LABELS "${_labels}")
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
