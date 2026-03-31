# SPECFEM++ CMake Configuration Summary
# Generates a formatted summary table of all configuration options

function(specfem_print_configuration_summary)
  # Color codes for console output
  if(NOT WIN32)
    string(ASCII 27 ESC)
    set(BOLD "${ESC}[1m")
    set(RESET "${ESC}[0m")
    set(CYAN "${ESC}[36m")
  else()
    set(BOLD "")
    set(RESET "")
    set(CYAN "")
  endif()

  # Build the summary string
  set(SUMMARY_STRING "")

  # Header
  string(APPEND SUMMARY_STRING "\n")
  string(APPEND SUMMARY_STRING "================================================================================\n")
  string(APPEND SUMMARY_STRING "                  SPECFEM++ CMAKE CONFIGURATION SUMMARY\n")
  string(APPEND SUMMARY_STRING "================================================================================\n")
  string(APPEND SUMMARY_STRING "\n")

  # ============================================================================
  # I/O OPTIONS
  # ============================================================================
  string(APPEND SUMMARY_STRING "I/O OPTIONS\n")
  string(APPEND SUMMARY_STRING "-----------\n")

  set(ADIOS2_STATUS "OFF")
  if(SPECFEM_ENABLE_ADIOS2)
    set(ADIOS2_STATUS "ON")
  endif()

  set(HDF5_STATUS "OFF")
  if(SPECFEM_ENABLE_HDF5)
    set(HDF5_STATUS "ON")
  endif()

  set(NPZ_STATUS "OFF")
  if(SPECFEM_ENABLE_NPZ)
    set(NPZ_STATUS "ON")
  endif()

  string(APPEND SUMMARY_STRING "  SPECFEM_ENABLE_ADIOS2                    | ${ADIOS2_STATUS}\n")
  string(APPEND SUMMARY_STRING "  SPECFEM_ENABLE_HDF5                      | ${HDF5_STATUS}\n")
  string(APPEND SUMMARY_STRING "  SPECFEM_ENABLE_NPZ                       | ${NPZ_STATUS}\n")
  string(APPEND SUMMARY_STRING "\n")

  # ============================================================================
  # FORCE INSTALL OPTIONS
  # ============================================================================
  string(APPEND SUMMARY_STRING "FORCE INSTALL OPTIONS\n")
  string(APPEND SUMMARY_STRING "---------------------\n")

  set(FORCE_ADIOS2 "OFF")
  if(SPECFEM_FORCE_INSTALL_ADIOS2)
    set(FORCE_ADIOS2 "ON")
  endif()

  set(FORCE_HDF5 "OFF")
  if(SPECFEM_FORCE_INSTALL_HDF5)
    set(FORCE_HDF5 "ON")
  endif()

  set(FORCE_NPZ "OFF")
  if(SPECFEM_FORCE_INSTALL_NPZ)
    set(FORCE_NPZ "ON")
  endif()

  set(FORCE_BOOST "OFF")
  if(SPECFEM_FORCE_INSTALL_BOOST)
    set(FORCE_BOOST "ON")
  endif()

  set(FORCE_KOKKOS "OFF")
  if(SPECFEM_FORCE_INSTALL_KOKKOS)
    set(FORCE_KOKKOS "ON")
  endif()

  set(FORCE_YAML "OFF")
  if(SPECFEM_FORCE_INSTALL_YAML)
    set(FORCE_YAML "ON")
  endif()

  string(APPEND SUMMARY_STRING "  SPECFEM_FORCE_INSTALL_ADIOS2             | ${FORCE_ADIOS2}\n")
  string(APPEND SUMMARY_STRING "  SPECFEM_FORCE_INSTALL_HDF5               | ${FORCE_HDF5}\n")
  string(APPEND SUMMARY_STRING "  SPECFEM_FORCE_INSTALL_NPZ                | ${FORCE_NPZ}\n")
  string(APPEND SUMMARY_STRING "  SPECFEM_FORCE_INSTALL_BOOST              | ${FORCE_BOOST}\n")
  string(APPEND SUMMARY_STRING "  SPECFEM_FORCE_INSTALL_KOKKOS             | ${FORCE_KOKKOS}\n")
  string(APPEND SUMMARY_STRING "  SPECFEM_FORCE_INSTALL_YAML               | ${FORCE_YAML}\n")
  string(APPEND SUMMARY_STRING "\n")

  # ============================================================================
  # BUILD & OPTIMIZATION
  # ============================================================================
  string(APPEND SUMMARY_STRING "BUILD & OPTIMIZATION\n")
  string(APPEND SUMMARY_STRING "--------------------\n")

  set(TESTS_STATUS "OFF")
  if(SPECFEM_ENABLE_TESTS)
    set(TESTS_STATUS "ON")
  endif()

  set(BENCHMARKS_STATUS "OFF")
  if(SPECFEM_ENABLE_BENCHMARKS)
    set(BENCHMARKS_STATUS "ON")
  endif()

  set(UNITY_BUILD_STATUS "OFF")
  if(SPECFEM_ENABLE_UNITY_BUILD)
    set(UNITY_BUILD_STATUS "ON")
  endif()

  string(APPEND SUMMARY_STRING "  SPECFEM_ENABLE_TESTS                     | ${TESTS_STATUS}\n")
  string(APPEND SUMMARY_STRING "  SPECFEM_ENABLE_BENCHMARKS                | ${BENCHMARKS_STATUS}\n")
  string(APPEND SUMMARY_STRING "  SPECFEM_ENABLE_UNITY_BUILD               | ${UNITY_BUILD_STATUS}\n")
  string(APPEND SUMMARY_STRING "\n")

  # ============================================================================
  # FEATURE FLAGS
  # ============================================================================
  string(APPEND SUMMARY_STRING "FEATURE FLAGS\n")
  string(APPEND SUMMARY_STRING "-------------\n")

  set(SIMD_STATUS "OFF")
  if(SPECFEM_ENABLE_SIMD)
    set(SIMD_STATUS "ON")
  endif()

  set(PROFILING_STATUS "OFF")
  if(SPECFEM_ENABLE_PROFILING)
    set(PROFILING_STATUS "ON")
  endif()

  set(DOUBLE_PRECISION_STATUS "ON")
  if(NOT SPECFEM_ENABLE_DOUBLE_PRECISION)
    set(DOUBLE_PRECISION_STATUS "OFF (Single Precision)")
  endif()

  set(MPI_STATUS "OFF")
  if(SPECFEM_ENABLE_MPI)
    set(MPI_STATUS "ON")
  endif()

  set(PYTHON_BINDING_STATUS "OFF")
  if(SPECFEM_ENABLE_PYTHON_BINDING)
    set(PYTHON_BINDING_STATUS "ON")
  endif()

  string(APPEND SUMMARY_STRING "  SPECFEM_ENABLE_SIMD                      | ${SIMD_STATUS}\n")
  string(APPEND SUMMARY_STRING "  SPECFEM_ENABLE_PROFILING                 | ${PROFILING_STATUS}\n")
  string(APPEND SUMMARY_STRING "  SPECFEM_ENABLE_DOUBLE_PRECISION          | ${DOUBLE_PRECISION_STATUS}\n")
  string(APPEND SUMMARY_STRING "  SPECFEM_ENABLE_MPI                       | ${MPI_STATUS}\n")
  string(APPEND SUMMARY_STRING "  SPECFEM_ENABLE_PYTHON_BINDING            | ${PYTHON_BINDING_STATUS}\n")
  string(APPEND SUMMARY_STRING "\n")

  # ============================================================================
  # GPU/CPU BUILD
  # ============================================================================
  string(APPEND SUMMARY_STRING "GPU/CPU BUILD\n")
  string(APPEND SUMMARY_STRING "-------------\n")

  set(BUILD_DEVICE ${CMAKE_BUILD_TYPE})
  if(NOT BUILD_DEVICE)
    set(BUILD_DEVICE "Release (default)")
  endif()

  string(APPEND SUMMARY_STRING "  CMAKE_BUILD_TYPE                         | ${BUILD_DEVICE}\n")
  string(APPEND SUMMARY_STRING "\n")

  # ============================================================================
  # INSTALLATION CONFIGURATION
  # ============================================================================
  string(APPEND SUMMARY_STRING "INSTALLATION CONFIGURATION\n")
  string(APPEND SUMMARY_STRING "----------------------------\n")

  string(APPEND SUMMARY_STRING "  CMAKE_INSTALL_PREFIX                     | ${CMAKE_INSTALL_PREFIX}\n")
  if(SPECFEM_BATCH_SIZE)
    string(APPEND SUMMARY_STRING "  SPECFEM_BATCH_SIZE                       | ${SPECFEM_BATCH_SIZE}\n")
  endif()
  string(APPEND SUMMARY_STRING "\n")

  # ============================================================================
  # COMPILER & PLATFORM
  # ============================================================================
  string(APPEND SUMMARY_STRING "COMPILER & PLATFORM\n")
  string(APPEND SUMMARY_STRING "-------------------\n")

  string(APPEND SUMMARY_STRING "  CMAKE_CXX_COMPILER_ID                    | ${CMAKE_CXX_COMPILER_ID}\n")
  string(APPEND SUMMARY_STRING "  CMAKE_CXX_COMPILER                       | ${CMAKE_CXX_COMPILER}\n")
  string(APPEND SUMMARY_STRING "  CMAKE_CXX_COMPILER_VERSION               | ${CMAKE_CXX_COMPILER_VERSION}\n")
  string(APPEND SUMMARY_STRING "  CMAKE_SYSTEM_NAME                        | ${CMAKE_SYSTEM_NAME}\n")
  string(APPEND SUMMARY_STRING "  CMAKE_SYSTEM_PROCESSOR                   | ${CMAKE_SYSTEM_PROCESSOR}\n")
  string(APPEND SUMMARY_STRING "\n")

  # ============================================================================
  # KOKKOS BACKENDS
  # ============================================================================
  string(APPEND SUMMARY_STRING "KOKKOS BACKENDS\n")
  string(APPEND SUMMARY_STRING "---------------\n")

  set(KOKKOS_CUDA_STATUS "OFF")
  if(Kokkos_ENABLE_CUDA)
    set(KOKKOS_CUDA_STATUS "ON")
  endif()

  set(KOKKOS_HIP_STATUS "OFF")
  if(Kokkos_ENABLE_HIP)
    set(KOKKOS_HIP_STATUS "ON")
  endif()

  set(KOKKOS_OPENMP_STATUS "OFF")
  if(Kokkos_ENABLE_OPENMP)
    set(KOKKOS_OPENMP_STATUS "ON")
  endif()

  set(KOKKOS_SERIAL_STATUS "OFF")
  if(Kokkos_ENABLE_SERIAL)
    set(KOKKOS_SERIAL_STATUS "ON")
  endif()

  string(APPEND SUMMARY_STRING "  Kokkos_ENABLE_CUDA                       | ${KOKKOS_CUDA_STATUS}\n")
  string(APPEND SUMMARY_STRING "  Kokkos_ENABLE_HIP                        | ${KOKKOS_HIP_STATUS}\n")
  string(APPEND SUMMARY_STRING "  Kokkos_ENABLE_OPENMP                     | ${KOKKOS_OPENMP_STATUS}\n")
  string(APPEND SUMMARY_STRING "  Kokkos_ENABLE_SERIAL                     | ${KOKKOS_SERIAL_STATUS}\n")
  string(APPEND SUMMARY_STRING "\n")

  # Footer
  string(APPEND SUMMARY_STRING "================================================================================\n")

  # Print to console
  message(STATUS "${SUMMARY_STRING}")

  # Write to file
  set(REPORT_FILE "${CMAKE_BINARY_DIR}/CMakeConfigurationReport.txt")
  file(WRITE "${REPORT_FILE}" "${SUMMARY_STRING}")
  message(STATUS "Configuration report saved to: ${REPORT_FILE}")

endfunction()
