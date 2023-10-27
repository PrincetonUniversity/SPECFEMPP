cmake_minimum_required(VERSION 3.10)

message("-- Building meshfem2D along with this project")
message("--     MESHFEM2D is build without MPI support")
message("--     MESHFEM2D is build without SCOTCH support")

unset(WITH_MPI)
unset(WITH_SCOTCH)

# Set the Fortran compiler
enable_language(Fortran)
enable_language(C)

set(FCFLAGS_f90 -g -O2 -fbacktrace)

## ========= MACROS =========
# define macro wrapper on custom command
macro(build_fortran_command module_name dependecies)
    get_filename_component(source_file_name ${module_name} NAME_WE)
    add_custom_command(
        OUTPUT ${module_name}
        COMMAND ${CMAKE_Fortran_COMPILER} ${FCFLAGS_f90} -c -o ${module_name} ${CMAKE_CURRENT_SOURCE_DIR}/${source_file_name}.f90
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${source_file_name}.f90 ${dependecies}
    )
endmacro()

macro(build_fortran_command_preprocess module_name dependecies)
    get_filename_component(source_file_name ${module_name} NAME_WE)
    add_custom_command(
        OUTPUT ${module_name}
        COMMAND ${CMAKE_Fortran_COMPILER} ${FCFLAGS_f90} -c -o ${module_name} ${CMAKE_CURRENT_SOURCE_DIR}/${source_file_name}.F90
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${source_file_name}.F90 ${dependecies}
    )
endmacro()

# define macro wrapper on custom command
macro(build_cc_command module_name)
    get_filename_component(source_file_name ${module_name} NAME_WE)
    add_custom_command(
        OUTPUT ${module_name}
        COMMAND ${CMAKE_C_COMPILER} ${CFLAGS} -c -o ${module_name} ${CMAKE_CURRENT_SOURCE_DIR}/${source_file_name}.c
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${source_file_name}.c config.h
    )
endmacro()
## ========= MACROS =========

set (SHARED_MODULE
    shared_par.shared_module.o
)

build_fortran_command(${SHARED_MODULE} ${CMAKE_CURRENT_SOURCE_DIR}/constants.h)

add_custom_target(shared_module ALL
    DEPENDS ${SHARED_MODULE}
)

set (UTILITIES
    command_line_arguments.mesh_utilities.o
    file_management.mesh_utilities.o
)

foreach(module_files ${UTILITIES})
    build_fortran_command(${module_files} shared_module)
endforeach()

add_custom_target(mesh_utilities ALL
    DEPENDS ${UTILITIES}
)

# shared objects
set(shared_OBJECTS
    define_shape_functions.shared.o;
    gll_library.shared.o;
    lagrange_poly.shared.o;
    read_interfaces_file.shared.o;
    read_material_table.shared.o;
    read_regions.shared.o;
    read_source_file.shared.o;
    read_value_parameters.shared.o;
    set_color_palette.shared.o;
    spline_routines.shared.o;
    write_VTK_data.shared.o
    CACHE INTERNAL "shared objects"
)

# Add a custom command to compile each object file
foreach(module_files ${shared_OBJECTS})
    build_fortran_command(${module_files} shared_module)
endforeach()

# Add a library target for the shared objects
add_custom_target(shared ALL
    DEPENDS ${shared_OBJECTS}
)

set(cc_OBJECTS
    force_ftz.cc.o;
    param_reader.cc.o
)


# Add a custom command to compile each object file

foreach(module_files ${cc_OBJECTS})
    build_cc_command(${module_files})
endforeach()

# Add a library target for the shared objects
add_custom_target(cc ALL
    DEPENDS ${cc_OBJECTS}
)

set(SHARED_MPI_LIBS
    read_parameter_file.shared.o;
    parallel.sharedmpi.o;
    exit_mpi.shared.o
    CACHE INTERNAL "shared MPI objects"
)

foreach(module_files ${SHARED_MPI_LIBS})
    build_fortran_command_preprocess(${module_files} shared_module)
endforeach()

# Add a library target for the shared objects
add_custom_target(shared_mpi ALL
    DEPENDS ${SHARED_MPI_LIBS}
)

set(MESHFEM_SHARED_OBJECTS
    ${SHARED_MODULE}
    ${UTILITIES}
    ${shared_OBJECTS}
    ${SHARED_MPI_LIBS}
    ${cc_OBJECTS}
)

set( MESHFEM_SHARED_MODULE
    meshfem2D_par.mesh_module.o
)

build_fortran_command(${MESHFEM_SHARED_MODULE} shared_module)

add_custom_target(mesh_module ALL
    DEPENDS ${MESHFEM_SHARED_MODULE}
)

set (MESHFEM_COMPUTE_ELEMENTS_LOAD
    compute_elements_load_par.mesh.o)

build_fortran_command(${MESHFEM_COMPUTE_ELEMENTS_LOAD} mesh_module)

add_custom_target(compute_elements_load ALL
    DEPENDS ${MESHFEM_COMPUTE_ELEMENTS_LOAD}
)

set( meshfem2D_OBJECTS
	determine_abs_surface.mesh.o
	determine_acoustic_surface.mesh.o
	get_node_number.mesh.o
	repartition_coupling.mesh.o
	rotate_mesh.mesh.o
	save_databases.mesh.o
	save_gnuplot_file.mesh.o
	save_stations_file.mesh.o
)

foreach(module_files ${meshfem2D_OBJECTS})
    build_fortran_command(${module_files} mesh_module)
endforeach()

add_custom_target(meshfem_mesh ALL
    DEPENDS ${meshfem2D_OBJECTS}
)

set( meshfem2D_PREPROCESSOR_OBJECTS
  	decompose_mesh.mesh.o
	metis_partitioning.mesh.o
	part_unstruct.mesh.o
	read_external_mesh_files.mesh.o
	read_mesh_files.mesh.o
  	scotch_partitioning.mesh.o
	meshfem2D.mesh.o
)

foreach(module_files ${meshfem2D_PREPROCESSOR_OBJECTS})
	build_fortran_command_preprocess(${module_files} compute_elements_load)
endforeach()

add_custom_target(mesh_mpi ALL
	DEPENDS ${meshfem2D_PREPROCESSOR_OBJECTS}
)

set( MESHFEM_OBJECTS
    ${MESHFEM_SHARED_MODULE}
    ${MESHFEM_COMPUTE_ELEMENTS_LOAD}
    ${meshfem2D_OBJECTS}
    ${meshfem2D_PREPROCESSOR_OBJECTS}
)

add_custom_command(
	OUTPUT meshfem2D
	COMMAND ${CMAKE_Fortran_COMPILER} ${FCFLAGS_f90} -o ${CMAKE_BINARY_DIR}/xmeshfem2D ${MESHFEM_OBJECTS} ${MESHFEM_SHARED_OBJECTS}
	DEPENDS ${MESHFEM_OBJECTS} ${MESHFEM_SHARED_OBJECTS}
)

add_custom_target(meshfem2D_exec ALL
	DEPENDS meshfem2D
)
