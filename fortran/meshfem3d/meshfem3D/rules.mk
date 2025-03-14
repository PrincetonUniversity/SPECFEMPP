#=====================================================================
#
#                         S p e c f e m 3 D
#                         -----------------
#
#     Main historical authors: Dimitri Komatitsch and Jeroen Tromp
#                              CNRS, France
#                       and Princeton University, USA
#                 (there are currently many more authors!)
#                           (c) October 2017
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
#=====================================================================


## compilation directories
S := ${S_TOP}/src/meshfem3D
$(meshfem3D_OBJECTS): S = ${S_TOP}/src/meshfem3D

#######################################

####
#### targets
####

meshfem3D_TARGETS = \
	$E/xmeshfem3D \
	$(EMPTY_MACRO)

meshfem3D_OBJECTS = \
	$O/calc_gll_points.mesh.o \
	$O/check_mesh_quality.mesh.o \
	$O/chunk_earth_mesh_mod.mesh.o \
	$O/compute_parameters.mesh.o \
	$O/create_meshfem_mesh.mesh.o \
	$O/create_CPML_regions.mesh.o \
	$O/create_interfaces_mesh.mesh.o \
	$O/create_visual_files.mesh.o \
	$O/define_subregions.mesh.o \
	$O/define_subregions_heuristic.mesh.o \
	$O/define_superbrick.mesh.o \
	$O/determine_cavity.mesh.o \
	$O/earth_chunk.mesh.o \
	$O/get_flags_boundaries.mesh.o \
	$O/get_MPI_cutplanes_eta.mesh.o \
	$O/get_MPI_cutplanes_xi.mesh.o \
	$O/meshfem3D.mesh.o \
	$O/meshfem3D_par.mesh_module.o \
	$O/get_wavefield_discontinuity.mesh.o \
	$O/read_mesh_parameter_file.mesh.o \
	$O/read_value_mesh_parameters.mesh.o \
	$O/save_databases.mesh.o \
	$O/save_databases_hdf5.mesh_hdf5.o \
	$O/store_boundaries.mesh.o \
	$O/store_coords.mesh.o \
	$(EMPTY_MACRO)

meshfem3D_MODULES = \
	$(FC_MODDIR)/constants_meshfem.$(FC_MODEXT) \
	$(FC_MODDIR)/manager_adios.$(FC_MODEXT) \
	$(FC_MODDIR)/meshfem_par.$(FC_MODEXT) \
	$(FC_MODDIR)/chunk_earth_mod.$(FC_MODEXT) \
	$(FC_MODDIR)/create_meshfem_par.$(FC_MODEXT) \
	$(EMPTY_MACRO)

meshfem3D_SHARED_OBJECTS = \
	$O/create_name_database.shared.o \
	$O/shared_par.shared_module.o \
	$O/adios_manager.shared_adios_module.o \
	$O/count_number_of_sources.shared.o \
	$O/exit_mpi.shared.o \
	$O/get_global.shared.o \
	$O/get_shape3D.shared.o \
	$O/gll_library.shared.o \
	$O/hdf5_manager.shared_hdf5_module.o \
	$O/hex_nodes.shared.o \
	$O/param_reader.cc.o \
	$O/read_parameter_file.shared.o \
	$O/read_topo_bathy_file.shared.o \
	$O/read_value_parameters.shared.o \
	$O/safe_alloc_mod.shared.o \
	$O/sort_array_coordinates.shared.o \
	$O/utm_geo.shared.o \
	$O/write_VTK_data.shared.o \
	$(EMPTY_MACRO)


# using ADIOS files
adios_meshfem3D_PREOBJECTS = \
	$O/adios_helpers_addons.shared_adios_cc.o \
	$O/adios_helpers_definitions.shared_adios.o \
	$O/adios_helpers_readers.shared_adios.o \
	$O/adios_helpers_writers.shared_adios.o \
	$O/adios_helpers.shared_adios.o \
	$(EMPTY_MACRO)

adios_meshfem3D_OBJECTS = \
	$O/save_databases_adios.mesh_adios.o \
	$(EMPTY_MACRO)

adios_meshfem3D_STUBS = \
	$O/adios_method_stubs.cc.o \
	$(EMPTY_MACRO)

# conditional adios linking
ifeq ($(ADIOS),yes)
meshfem3D_OBJECTS += $(adios_meshfem3D_OBJECTS)
meshfem3D_SHARED_OBJECTS += $(adios_meshfem3D_PREOBJECTS)
else ifeq ($(ADIOS2),yes)
meshfem3D_OBJECTS += $(adios_meshfem3D_OBJECTS)
meshfem3D_SHARED_OBJECTS += $(adios_meshfem3D_PREOBJECTS)
else
meshfem3D_SHARED_OBJECTS += $(adios_meshfem3D_STUBS)
endif

# objects for the pure Fortran version
XMESHFEM_OBJECTS = \
	$(meshfem3D_OBJECTS) $(meshfem3D_SHARED_OBJECTS) \
	$(COND_MPI_OBJECTS)


#######################################


####
#### rules for executables
####

mesh: $(meshfem3D_TARGETS)

meshfem3D: xmeshfem3D
xmeshfem3D: $E/xmeshfem3D

# rules for the pure Fortran version
$E/xmeshfem3D: $(XMESHFEM_OBJECTS)
	@echo ""
	@echo "building xmeshfem3D"
	@echo ""
	${FCLINK} -o ${E}/xmeshfem3D $(XMESHFEM_OBJECTS) $(MPILIBS)
	@echo ""



#######################################


###
### Module dependencies
###

# Version file
$O/meshfem3D.mesh.o: ${SETUP}/version.fh

$O/meshfem3D.mesh.o: $O/chunk_earth_mesh_mod.mesh.o
$O/determine_cavity.mesh.o: $O/create_meshfem_mesh.mesh.o

## adios
$O/save_databases_adios.mesh_adios.o: $O/safe_alloc_mod.shared.o $O/adios_manager.shared_adios_module.o
$O/create_meshfem_mesh.mesh.o: $O/adios_manager.shared_adios_module.o

####
#### rule to build each .o file below
####

$O/%.mesh_module.o: $S/%.f90 $O/shared_par.shared_module.o $S/constants_meshfem3D.h
	${FCCOMPILE_CHECK} ${FCFLAGS_f90} -c -o $@ $<

$O/%.mesh.o: $S/%.f90 $O/shared_par.shared_module.o $O/meshfem3D_par.mesh_module.o
	${FCCOMPILE_CHECK} ${FCFLAGS_f90} -c -o $@ $<

$O/%.mesh.o: $S/%.F90 $O/shared_par.shared_module.o $O/meshfem3D_par.mesh_module.o
	${FCCOMPILE_CHECK} ${FCFLAGS_f90} -c -o $@ $<


###
### ADIOS compilation
###

$O/%.mesh_adios.o: $S/%.F90 $O/shared_par.shared_module.o $O/meshfem3D_par.mesh_module.o $O/adios_helpers.shared_adios.o
	${FCCOMPILE_CHECK} ${FCFLAGS_f90} -c -o $@ $<

$O/%.mesh_adios.o: $S/%.f90 $O/shared_par.shared_module.o $O/meshfem3D_par.mesh_module.o $O/adios_helpers.shared_adios.o
	${FCCOMPILE_CHECK} ${FCFLAGS_f90}  -c -o $@ $<

$O/%.mesh_noadios.o: $S/%.F90 $O/meshfem3D_par.mesh_module.o
	${FCCOMPILE_CHECK} ${FCFLAGS_f90} -c -o $@ $<

$O/%.mesh_noadios.o: $S/%.f90 $O/meshfem3D_par.mesh_module.o
	${FCCOMPILE_CHECK} ${FCFLAGS_f90} -c -o $@ $<


## HDF5 file i/o

$O/%.mesh_hdf5.o: $S/%.F90 $O/shared_par.shared_module.o $O/meshfem3D_par.mesh_module.o $O/hdf5_manager.shared_hdf5_module.o
	${FCCOMPILE_CHECK} ${FCFLAGS_f90} -c -o $@ $<

$O/%.mesh_hdf5.o: $S/%.f90 $O/shared_par.shared_module.o $O/meshfem3D_par.mesh_module.o $O/hdf5_manager.shared_hdf5_module.o
	${FCCOMPILE_CHECK} ${FCFLAGS_f90}  -c -o $@ $<
