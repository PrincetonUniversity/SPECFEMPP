cmake_minimum_required(VERSION 3.17.5)

# I need to automate this process

configure_file(elastic_domain/cpu/specfem_config.yaml.in ${CMAKE_SOURCE_DIR}/tests/regression-tests/elastic_domain/cpu/specfem_config.yaml)
configure_file(elastic_domain/gpu/specfem_config.yaml.in ${CMAKE_SOURCE_DIR}/tests/regression-tests/elastic_domain/gpu/specfem_config.yaml)

add_executable(
    compare_regression_results
    compare_regression_results.cpp
)

target_link_libraries(
    compare_regression_results
    Boost::program_options
    yaml-cpp
    Kokkos::kokkos
)
