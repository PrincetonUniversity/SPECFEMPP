cmake_minimum_required(VERSION 3.17.5)

# I need to automate this process

configure_file(elastic_domain/cpu/specfem_config.yaml.in ${CMAKE_SOURCE_DIR}/tests/regression-tests/elastic_domain/cpu/specfem_config.yaml)
configure_file(elastic_domain/gpu/specfem_config.yaml.in ${CMAKE_SOURCE_DIR}/tests/regression-tests/elastic_domain/gpu/specfem_config.yaml)
