# ==============================================================================
# Helper macros for test registration with optional install-dir support
# These macros support both in-tree builds and external TEST_OUTPUT_DIR
# ==============================================================================

# Write the path-fix script used when running tests from TEST_OUTPUT_DIR
macro(specfem_write_copy_test_cmake_script)
    if(NOT SPECFEM_TESTDIR_DEFAULT)
        file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/copy_test_cmake.cmake [=[
file(READ "${INPUT_FILE}" content)
string(REPLACE "${OLD_PATH}" "${NEW_PATH}" content "${content}")
file(WRITE "${OUTPUT_FILE}" "${content}")
]=])
    endif()
endmacro()

# Register a single test target with binary copy, discovery, and optional path-fix
# Usage: specfem_register_test_target(test_executable_name)
macro(specfem_register_test_target TARGET)
    # Copy test binary to TEST_OUTPUT_DIR after build
    add_custom_command(TARGET ${TARGET} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${TEST_OUTPUT_DIR}
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:${TARGET}>
            ${TEST_OUTPUT_DIR}/$<TARGET_FILE_NAME:${TARGET}>
        COMMENT "Moving ${TARGET} to ${TEST_OUTPUT_DIR}"
    )

    # When MPI is enabled, run serial tests under mpirun -n 1 so MPI initializes correctly
    if(SPECFEM_ENABLE_MPI)
        set_target_properties(${TARGET} PROPERTIES
            CROSSCOMPILING_EMULATOR "${MPIEXEC_EXECUTABLE};${MPIEXEC_NUMPROC_FLAG};1"
        )
    endif()

    # Discover tests with MPI-specific configuration
    # Note: PROCESSORS property is set to 1 (runs via mpirun -n 1 via CROSSCOMPILING_EMULATOR)
    # Unlike mpi.cmake, RUN_SERIAL is not used here because these are serial tests running under MPI initialization
    # Discover tests from TEST_OUTPUT_DIR
    if(SPECFEM_ENABLE_MPI)
        gtest_discover_tests(${TARGET}
            DISCOVERY_MODE POST_BUILD
            DISCOVERY_TIMEOUT 300
            WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
            PROPERTIES
                PROCESSORS 1
        )
    else()
        gtest_discover_tests(${TARGET}
            DISCOVERY_MODE POST_BUILD
            DISCOVERY_TIMEOUT 300
            WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
        )
    endif()

    # When TEST_OUTPUT_DIR is external, copy and fix the discovery .cmake files
    if(NOT SPECFEM_TESTDIR_DEFAULT)
        add_custom_command(TARGET ${TARGET} POST_BUILD
            COMMAND ${CMAKE_COMMAND}
                -DINPUT_FILE=${CMAKE_CURRENT_BINARY_DIR}/${TARGET}[1]_tests.cmake
                -DOUTPUT_FILE=${TEST_OUTPUT_DIR}/${TARGET}_tests.cmake
                -DOLD_PATH=${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
                -DNEW_PATH=${TEST_OUTPUT_DIR}
                -P ${CMAKE_CURRENT_BINARY_DIR}/copy_test_cmake.cmake
            COMMENT "Copying ${TARGET} test discovery files to ${TEST_OUTPUT_DIR}"
        )
    endif()
endmacro()

# Finalize test setup: generate CTestTestfile.cmake and setup data dirs
# Usage: specfem_finalize_test_targets(TEST_TARGETS_LIST_VAR LINK_DIRS_LIST_VAR)
macro(specfem_finalize_test_targets TEST_TARGETS_VAR LINK_DIRS_VAR)
    # When TEST_OUTPUT_DIR is external, generate portable CTestTestfile.cmake
    if(NOT SPECFEM_TESTDIR_DEFAULT)
        set(_ctest_content "")
        foreach(test_target IN LISTS ${TEST_TARGETS_VAR})
            file(WRITE ${TEST_OUTPUT_DIR}/${test_target}_include.cmake
                "if(EXISTS \"${TEST_OUTPUT_DIR}/${test_target}_tests.cmake\")\n"
                "  include(\"${TEST_OUTPUT_DIR}/${test_target}_tests.cmake\")\n"
                "else()\n"
                "  add_test(${test_target}_NOT_BUILT ${test_target}_NOT_BUILT)\n"
                "endif()\n"
            )
            string(APPEND _ctest_content "include(\"${TEST_OUTPUT_DIR}/${test_target}_include.cmake\")\n")
        endforeach()
        file(WRITE ${TEST_OUTPUT_DIR}/CTestTestfile.cmake "${_ctest_content}")
    endif()

    # Setup test data directories: symlinks for in-tree, install for external
    foreach(dir_name IN LISTS ${LINK_DIRS_VAR})
        if(SPECFEM_TESTDIR_DEFAULT)
            file(CREATE_LINK
                ${CMAKE_CURRENT_SOURCE_DIR}/${dir_name}
                ${CMAKE_CURRENT_BINARY_DIR}/${dir_name}
                SYMBOLIC)
        else()
            install(DIRECTORY ${dir_name} DESTINATION ${TEST_OUTPUT_DIR} COMPONENT tests)
        endif()
    endforeach()

    # Create install target for external TEST_OUTPUT_DIR
    if (NOT SPECFEM_TESTDIR_DEFAULT)
        add_custom_target(install_test_data ALL
            COMMAND ${CMAKE_COMMAND} --install . --prefix ${TEST_OUTPUT_DIR} --component tests
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/tests/unit-tests
            COMMENT "Installing test data to ${TEST_OUTPUT_DIR}"
        )
    endif()
endmacro()


# ==============================================================================
# Helper function to register an MPI test with configurable processor count
# and test properties
# ==============================================================================
# Usage: add_mpi_test(TARGET NUM_PROCESSES)
# Example: add_mpi_test(mesh_mpi_dim3_tests 4)
# Uses gtest_discover_tests with POST_BUILD discovery mode and
# CROSSCOMPILING_EMULATOR to run the binary via the MPI launcher.
function(add_mpi_test TARGET NUM_PROCESSES)
    # Copy test binary to TEST_OUTPUT_DIR after build
    add_custom_command(TARGET ${TARGET} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${TEST_OUTPUT_DIR}
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:${TARGET}>
            ${TEST_OUTPUT_DIR}/$<TARGET_FILE_NAME:${TARGET}>
        COMMENT "Moving ${TARGET} to ${TEST_OUTPUT_DIR}"
    )

    # CROSSCOMPILING_EMULATOR prefixes test execution with the MPI launcher.
    set_target_properties(${TARGET} PROPERTIES
        CROSSCOMPILING_EMULATOR "${MPIEXEC_EXECUTABLE};${MPIEXEC_NUMPROC_FLAG};${NUM_PROCESSES}"
    )

    # Discover tests from TEST_OUTPUT_DIR using POST_BUILD mode.
    # CROSSCOMPILING_EMULATOR is used to run the binary with the MPI launcher.
    gtest_discover_tests(${TARGET}
        DISCOVERY_MODE POST_BUILD
        DISCOVERY_TIMEOUT 300
        WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
        PROPERTIES
            PROCESSORS ${NUM_PROCESSES}
            TIMEOUT 300
            RUN_SERIAL ON
    )

    # When TEST_OUTPUT_DIR is external, copy and fix the discovery .cmake files
    if(NOT SPECFEM_TESTDIR_DEFAULT)
        add_custom_command(TARGET ${TARGET} POST_BUILD
            COMMAND ${CMAKE_COMMAND}
                -DINPUT_FILE=${CMAKE_CURRENT_BINARY_DIR}/${TARGET}[1]_tests.cmake
                -DOUTPUT_FILE=${TEST_OUTPUT_DIR}/${TARGET}_tests.cmake
                -DOLD_PATH=${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
                -DNEW_PATH=${TEST_OUTPUT_DIR}
                -P ${CMAKE_CURRENT_BINARY_DIR}/copy_test_cmake.cmake
            COMMENT "Copying ${TARGET} test discovery files to ${TEST_OUTPUT_DIR}"
        )
    endif()
endfunction()
