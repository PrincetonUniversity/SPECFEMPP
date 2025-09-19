
CMAKE_MINIMUM_REQUIRED(VERSION 3.17.5)



CMAKE_MINIMUM_REQUIRED(VERSION 3.17.5)

# This basically jsut gets the example name
get_filename_component(CURRENT_DIR_NAME "${CMAKE_CURRENT_SOURCE_DIR}" NAME)


if (NOT DEFINED SPECFEM_BENCHMARKS_BUILD_DIR)
    set(SPECFEM_BENCHMARKS_BUILD_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../build)
endif()

set(CURRENT_BENCHMARK_BUILD_DIR "${SPECFEM_BENCHMARKS_BUILD_DIR}/dim3/${CURRENT_DIR_NAME}")

# Copy static files into the benchmarks.
file(
  COPY
    ${CMAKE_CURRENT_SOURCE_DIR}/README.md
    ${CMAKE_CURRENT_SOURCE_DIR}/DATA
    ${CMAKE_CURRENT_SOURCE_DIR}/force.yaml
  DESTINATION ${CURRENT_BENCHMARK_BUILD_DIR}
)

# Configure files directly into the cmake benchmarks_build!
configure_file(CMakeFiles/Mesh_Par_file.in ${CURRENT_BENCHMARK_BUILD_DIR}/DATA/meshfem3D_files/Mesh_Par_file)
configure_file(CMakeFiles/interfaces.txt.in ${CURRENT_BENCHMARK_BUILD_DIR}/DATA/meshfem3D_files/interfaces.txt)
configure_file(CMakeFiles/Par_File.in ${CURRENT_BENCHMARK_BUILD_DIR}/DATA/Par_File)
configure_file(CMakeFiles/specfem_config.yaml.in ${CURRENT_BENCHMARK_BUILD_DIR}/specfem_config.yaml)

# Mkdir CAN BE REMOVED ONCE WE HAVE A SNAKEFILE WITH FULL EXAMPLE
file(MAKE_DIRECTORY ${CURRENT_BENCHMARK_BUILD_DIR}/OUTPUT_FILES/DATABASES_MPI)
file(MAKE_DIRECTORY ${CURRENT_BENCHMARK_BUILD_DIR}/OUTPUT_FILES/results)
