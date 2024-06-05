
CMAKE_MINIMUM_REQUIRED(VERSION 3.17.5)

configure_file(specfem_config.yaml.in ${CMAKE_SOURCE_DIR}/examples/Tromp_2005/specfem_config.yaml)
configure_file(Par_File.in ${CMAKE_SOURCE_DIR}/examples/Tromp_2005/Par_File)
configure_file(source.yaml.in ${CMAKE_SOURCE_DIR}/examples/Tromp_2005/source.yaml)
