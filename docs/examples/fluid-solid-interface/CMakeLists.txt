
CMAKE_MINIMUM_REQUIRED(VERSION 3.17.5)

configure_file(specfem_config.yaml.in ${CMAKE_SOURCE_DIR}/examples/fluid-solid-interface/specfem_config.yaml)
configure_file(Par_File.in ${CMAKE_SOURCE_DIR}/examples/fluid-solid-interface/Par_File)
