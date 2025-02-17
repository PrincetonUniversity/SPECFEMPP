
CMAKE_MINIMUM_REQUIRED(VERSION 3.17.5)


configure_file(config/Mesh_Par_file.in ${CMAKE_CURRENT_SOURCE_DIR}/DATA/meshfem3D_files/Mesh_Par_file)
configure_file(config/interfaces.txt.in ${CMAKE_CURRENT_SOURCE_DIR}/DATA/meshfem3D_files/interfaces.txt)
configure_file(config/Par_File.in ${CMAKE_CURRENT_SOURCE_DIR}/Par_File)
configure_file(config/specfem_config.yaml.in ${CMAKE_CURRENT_SOURCE_DIR}/specfem_config.yaml)

# Mkdir
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/OUTPUT_FILES/DATABASES_MPI)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/OUTPUT_FILES/results)
