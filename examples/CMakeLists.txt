cmake_minimum_required(VERSION 3.17.5)

configure_file(homogeneous-medium-flat-topography/specfem_config.yaml.in ${CMAKE_SOURCE_DIR}/examples/homogeneous-medium-flat-topography/specfem_config.yaml)
configure_file(fluid-solid-interface/specfem_config.yaml.in ${CMAKE_SOURCE_DIR}/examples/fluid-solid-interface/specfem_config.yaml)
