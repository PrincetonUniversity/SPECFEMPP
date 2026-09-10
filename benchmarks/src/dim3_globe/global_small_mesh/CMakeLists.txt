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
    ${CMAKE_CURRENT_SOURCE_DIR}/check_database.py
    ${CMAKE_CURRENT_SOURCE_DIR}/DATA
  DESTINATION ${CURRENT_BENCHMARK_BUILD_DIR}
)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/CMakeFiles/Snakefile.in
  ${CURRENT_BENCHMARK_BUILD_DIR}/Snakefile
  @ONLY
)

file(MAKE_DIRECTORY ${CURRENT_BENCHMARK_BUILD_DIR}/DATABASES_MPI)
file(MAKE_DIRECTORY ${CURRENT_BENCHMARK_BUILD_DIR}/OUTPUT_FILES)

if(SPECFEM_BUILD_MESHFEM3D_GLOBE)
  add_custom_target(benchmark_dim3_globe_global_small_mesh
    COMMAND ${CMAKE_COMMAND} -E env
            PATH=$<TARGET_FILE_DIR:xmeshfem3D_globe>:$ENV{PATH}
            ${CMAKE_COMMAND} -E chdir ${CURRENT_BENCHMARK_BUILD_DIR}
            ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 6
            $<TARGET_FILE:xmeshfem3D_globe>
    # validates the thin SPECFEM++ mesh database (SPECFEMPP_DATABASE = .true. in the
    # Par_file): record framing, node/element consistency, CSR symmetry, CMB/ICB
    # welding, boundary face counts, cross-rank MPI interface agreement, and that
    # the model config block is present and identical on every rank
    COMMAND ${CMAKE_COMMAND} -E chdir ${CURRENT_BENCHMARK_BUILD_DIR}
            uv run --project ${PROJECT_SOURCE_DIR} --group examples
            python check_database.py DATABASES_MPI
    DEPENDS xmeshfem3D_globe
    WORKING_DIRECTORY ${CURRENT_BENCHMARK_BUILD_DIR}
    COMMENT "Generating dim3_globe global_small_mesh with xmeshfem3D_globe"
    VERBATIM
  )
endif()
