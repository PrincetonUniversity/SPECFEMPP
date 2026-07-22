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
    ${CMAKE_CURRENT_SOURCE_DIR}/scripts
  DESTINATION ${CURRENT_BENCHMARK_BUILD_DIR}
)

# This benchmark is MPI-only (it is registered from benchmarks/src/CMakeLists.txt
# only when SPECFEM_ENABLE_MPI is ON), so there is a single set of templates.
configure_file(CMakeFiles/export_mesh.py.in ${CURRENT_BENCHMARK_BUILD_DIR}/scripts/export_mesh.py)
configure_file(CMakeFiles/specfem_config.yaml.in ${CURRENT_BENCHMARK_BUILD_DIR}/specfem_config.yaml)

if (SPECFEM_ENABLE_MPI)
  set(NPROC 4)
else()
  set(NPROC 1)
endif()

# These two files end up being different for serial and MPI.
configure_file(CMakeFiles/Par_file.in ${CURRENT_BENCHMARK_BUILD_DIR}/Par_file)
configure_file(CMakeFiles/Snakefile.in ${CURRENT_BENCHMARK_BUILD_DIR}/Snakefile)

# Link the reference seismograms (committed in the source tree) into the build
# directory so plot_seismograms.py can find them next to the computed traces.
# file(CREATE_LINK <original> <linkname>): the link is created AT <linkname>
# (build dir) pointing TO <original> (source dir).
file(
  CREATE_LINK
    ${CMAKE_CURRENT_SOURCE_DIR}/reference_seismograms
    ${CURRENT_BENCHMARK_BUILD_DIR}/reference_seismograms
  SYMBOLIC
)
