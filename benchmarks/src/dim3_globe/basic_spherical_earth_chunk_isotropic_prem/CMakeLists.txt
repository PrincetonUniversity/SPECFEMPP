cmake_minimum_required(VERSION 3.17.5)

get_filename_component(CURRENT_DIR_NAME "${CMAKE_CURRENT_SOURCE_DIR}" NAME)

if(NOT DEFINED SPECFEM_BENCHMARKS_BUILD_DIR)
  set(SPECFEM_BENCHMARKS_BUILD_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../build)
endif()

set(CURRENT_BENCHMARK_BUILD_DIR
    "${SPECFEM_BENCHMARKS_BUILD_DIR}/dim3_globe/${CURRENT_DIR_NAME}")

file(
  COPY
    ${CMAKE_CURRENT_SOURCE_DIR}/README.md
    ${CMAKE_CURRENT_SOURCE_DIR}/DATA
    ${CMAKE_CURRENT_SOURCE_DIR}/plot_seismograms.py
  DESTINATION ${CURRENT_BENCHMARK_BUILD_DIR}
)

file(
  CREATE_LINK
    ${CMAKE_CURRENT_SOURCE_DIR}/reference_seismograms
    ${CURRENT_BENCHMARK_BUILD_DIR}/reference_seismograms
    SYMBOLIC
)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/CMakeFiles/Snakefile.in
  ${CURRENT_BENCHMARK_BUILD_DIR}/Snakefile
  @ONLY
)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/CMakeFiles/specfem_config.yaml.in
  ${CURRENT_BENCHMARK_BUILD_DIR}/specfem_config.yaml
  @ONLY
)

file(MAKE_DIRECTORY ${CURRENT_BENCHMARK_BUILD_DIR}/DATABASES_MPI)
file(MAKE_DIRECTORY ${CURRENT_BENCHMARK_BUILD_DIR}/OUTPUT_FILES)
file(MAKE_DIRECTORY ${CURRENT_BENCHMARK_BUILD_DIR}/OUTPUT_FILES/results)

# single chunk, single rank: the serial mesher is run directly
add_custom_target(benchmark_dim3_globe_basic_spherical_earth_chunk_isotropic_prem
  COMMAND ${CMAKE_COMMAND} -E chdir ${CURRENT_BENCHMARK_BUILD_DIR}
          $<TARGET_FILE:xmeshfem3D_globe>
  DEPENDS xmeshfem3D_globe
  WORKING_DIRECTORY ${CURRENT_BENCHMARK_BUILD_DIR}
  COMMENT "Generating dim3_globe basic_spherical_earth_chunk_isotropic_prem with xmeshfem3D_globe"
  VERBATIM
)
