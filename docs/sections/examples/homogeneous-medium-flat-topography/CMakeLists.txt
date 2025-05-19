
CMAKE_MINIMUM_REQUIRED(VERSION 3.17.5)

configure_file(Par_File.in ${CMAKE_SOURCE_DIR}/examples/homogeneous-medium-flat-topography/Par_File)
configure_file(specfem_config.yaml.in ${CMAKE_SOURCE_DIR}/examples/homogeneous-medium-flat-topography/specfem_config.yaml)
