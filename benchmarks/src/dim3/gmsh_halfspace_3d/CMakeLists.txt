
CMAKE_MINIMUM_REQUIRED(VERSION 3.17.5)

get_filename_component(CURRENT_DIR_NAME "${CMAKE_CURRENT_SOURCE_DIR}" NAME)

if (NOT DEFINED SPECFEM_BENCHMARKS_BUILD_DIR)
    set(SPECFEM_BENCHMARKS_BUILD_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../build)
endif()

set(CURRENT_BENCHMARK_BUILD_DIR "${SPECFEM_BENCHMARKS_BUILD_DIR}/dim3/${CURRENT_DIR_NAME}")

# Copy static files into the build directory
file(
  COPY
    ${CMAKE_CURRENT_SOURCE_DIR}/README.md
    ${CMAKE_CURRENT_SOURCE_DIR}/DATA
    ${CMAKE_CURRENT_SOURCE_DIR}/sources.yaml
    ${CMAKE_CURRENT_SOURCE_DIR}/create_mesh.py
    ${CMAKE_CURRENT_SOURCE_DIR}/plot_seismograms.py
  DESTINATION ${CURRENT_BENCHMARK_BUILD_DIR}
)

configure_file(CMakeFiles/export_mesh.py.in ${CURRENT_BENCHMARK_BUILD_DIR}/export_mesh.py)

if (SPECFEM_ENABLE_MPI)
  configure_file(CMakeFiles/Par_file_mpi.in ${CURRENT_BENCHMARK_BUILD_DIR}/Par_file)
  configure_file(CMakeFiles/specfem_config_mpi.yaml.in ${CURRENT_BENCHMARK_BUILD_DIR}/specfem_config.yaml)
  configure_file(CMakeFiles/Snakefile_mpi.in ${CURRENT_BENCHMARK_BUILD_DIR}/Snakefile)
else ()
  configure_file(CMakeFiles/Par_file.in ${CURRENT_BENCHMARK_BUILD_DIR}/Par_file)
  configure_file(CMakeFiles/specfem_config.yaml.in ${CURRENT_BENCHMARK_BUILD_DIR}/specfem_config.yaml)
  configure_file(CMakeFiles/Snakefile.in ${CURRENT_BENCHMARK_BUILD_DIR}/Snakefile)
endif()

file(MAKE_DIRECTORY ${CURRENT_BENCHMARK_BUILD_DIR}/MESH)
file(MAKE_DIRECTORY ${CURRENT_BENCHMARK_BUILD_DIR}/DATABASES_MPI)
file(MAKE_DIRECTORY ${CURRENT_BENCHMARK_BUILD_DIR}/OUTPUT_FILES/results)
