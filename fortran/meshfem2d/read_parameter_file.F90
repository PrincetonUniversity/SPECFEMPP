!========================================================================
!
!                            S P E C F E M 2 D
!                            -----------------
!
!     Main historical authors: Dimitri Komatitsch and Jeroen Tromp
!                              CNRS, France
!                       and Princeton University, USA
!                 (there are currently many more authors!)
!                           (c) October 2017
!
! This software is a computer program whose purpose is to solve
! the two-dimensional viscoelastic anisotropic or poroelastic wave equation
! using a spectral-element method (SEM).
!
! This program is free software; you can redistribute it and/or modify
! it under the terms of the GNU General Public License as published by
! the Free Software Foundation; either version 3 of the License, or
! (at your option) any later version.
!
! This program is distributed in the hope that it will be useful,
! but WITHOUT ANY WARRANTY; without even the implied warranty of
! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
! GNU General Public License for more details.
!
! You should have received a copy of the GNU General Public License along
! with this program; if not, write to the Free Software Foundation, Inc.,
! 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
!
! The full text of the license is available in file "LICENSE".
!
!========================================================================

subroutine read_parameter_file(imesher,BROADCAST_AFTER_READ)

! reads in DATA/Par_file

   use constants, only: IMAIN,myrank
   use shared_parameters

   implicit none

   integer, intent(in) :: imesher
   logical, intent(in) :: BROADCAST_AFTER_READ

   ! initializes
   call read_parameter_file_init()

   ! only main process reads in Par_file
   if (myrank == 0) then

      ! user output
      write(IMAIN,*) 'Reading the parameter file...'
      write(IMAIN,*)
      call flush_IMAIN()

      ! opens file Par_file
      call open_parameter_file()

      ! reads only parameters (without receiver-line section, material tables or region definitions)
      call read_parameter_file_only()

      ! user output
      write(IMAIN,*) 'Title of the simulation: ',trim(title)
      write(IMAIN,*)
      call flush_IMAIN()

      ! reads receiver lines
      if (imesher == 1) then
         ! user output
         write(IMAIN,*) 'Receiver lines:'
         write(IMAIN,*) '  Nb of line sets = ',nreceiversets
         write(IMAIN,*)
         call flush_IMAIN()
      endif
      call read_parameter_file_receiversets()

      ! only mesher needs to reads this
      if (imesher == 1) then
         ! reads material definitions
         call read_material_table()

         ! mesher reads in internal region table for setting up mesh elements
         if (.not. read_external_mesh) then
            ! internal meshing
            ! user output
            write(IMAIN,*)
            write(IMAIN,*) 'Mesh from internal meshing:'
            write(IMAIN,*)
            call flush_IMAIN()

            ! reads interface definitions from interface file (we need to have nxread & nzread value for checking regions)
            call read_interfaces_file()

            ! internal meshing
            nx_elem_internal = nxread
            nz_elem_internal = nzread

            ! setup mesh array
            ! multiply by 2 if elements have 9 nodes
            if (NGNOD == 9) then
               nx_elem_internal = nx_elem_internal * 2
               nz_elem_internal = nz_elem_internal * 2
               nz_layer(:) = nz_layer(:) * 2
            endif

            ! total number of elements
            nelmnts = nxread * nzread

            ! reads material regions defined in Par_file
            call read_regions()
         endif
      endif

      ! closes file Par_file
      call close_parameter_file()
   endif

   ! main process broadcasts to all
   ! note: this is only needed at the moment for the solver to setup a simulation run
   if (BROADCAST_AFTER_READ) then

      call bcast_all_singlei(NPROC)
      call bcast_all_singlei(PARTITIONING_TYPE)
      call bcast_all_singlei(NGNOD)

      ! receivers
      call bcast_all_singlel(use_existing_STATIONS)
      call bcast_all_singlei(nreceiversets)
      call bcast_all_singledp(anglerec)
      call bcast_all_singlel(rec_normal_to_surface)

      ! velocity and density models
      call bcast_all_singlei(nbmodels)
      call bcast_all_string(TOMOGRAPHY_FILE)
      call bcast_all_singlel(read_external_mesh)

      if (read_external_mesh) then
         call bcast_all_string(mesh_file)
         call bcast_all_string(nodes_coords_file)
         call bcast_all_string(materials_file)
         call bcast_all_string(free_surface_file)
         call bcast_all_string(axial_elements_file)
         call bcast_all_string(absorbing_surface_file)
         call bcast_all_string(acoustic_forcing_surface_file)
         call bcast_all_string(absorbing_cpml_file)
         call bcast_all_string(tangential_detection_curve_file)
         call bcast_all_string(nonconforming_adjacencies_file)
      else
         call bcast_all_string(interfaces_filename)
         call bcast_all_singledp(xmin_param)
         call bcast_all_singledp(xmax_param)
         call bcast_all_singlei(nx_param)

         call bcast_all_singlel(STACEY_ABSORBING_CONDITIONS)
         call bcast_all_singlel(absorbbottom)
         call bcast_all_singlel(absorbright)
         call bcast_all_singlel(absorbtop)
         call bcast_all_singlel(absorbleft)
         call bcast_all_singlei(nbregions)
      endif
   endif

   ! derive additional settings/flags based on input parameters
   call read_parameter_file_derive_flags()

   ! user output
   if (myrank == 0) then
      write(IMAIN,*) 'Parameter file successfully read '
      write(IMAIN,*)
      call flush_IMAIN()
   endif

end subroutine read_parameter_file

!
!-------------------------------------------------------------------------------------------------
!

subroutine read_parameter_file_init()

! initializes the variables

   use shared_parameters

   implicit none

   ! external meshing
   mesh_file = ''
   nodes_coords_file = ''
   materials_file = ''
   free_surface_file = ''
   axial_elements_file = ''
   absorbing_surface_file = ''
   acoustic_forcing_surface_file = ''
   absorbing_cpml_file = ''
   tangential_detection_curve_file = ''
   nonconforming_adjacencies_file = ''
   should_read_nonconforming_adjacencies_file = .false.

   ! internal meshing
   interfaces_filename = ''
   xmin_param = 0.d0
   xmax_param = 0.d0
   nx_param = 0

   STACEY_ABSORBING_CONDITIONS = .false.
   absorbbottom = .false.
   absorbright = .false.
   absorbtop = .false.
   absorbleft = .false.

   nbregions = 0

end subroutine read_parameter_file_init

!
!-------------------------------------------------------------------------------------------------
!

subroutine read_parameter_file_only()

! reads only parameters without receiver-line section and material tables

   use constants, only: IMAIN,myrank
   use shared_parameters

   implicit none

   ! local parameters
   integer :: i,irange
   logical :: some_parameters_missing_from_Par_file

   integer, external :: err_occurred

!! DK DK to detect discontinued parameters
   double precision :: f0_attenuation

   !--------------------------------------------------------------------
   !
   ! simulation input parameters
   !
   !--------------------------------------------------------------------

   some_parameters_missing_from_Par_file = .false.

   ! read file names and path for output
   call read_value_string_p(title, 'title')
   if (err_occurred() /= 0) then
      some_parameters_missing_from_Par_file = .true.
      write(*,'(a)') 'title                           = Title of my simulation'
      write(*,*)
   endif

   ! read info about partitioning
   call read_value_integer_p(NPROC, 'NPROC')
   if (err_occurred() /= 0) then
      some_parameters_missing_from_Par_file = .true.
      write(*,'(a)') 'NPROC                           = 1'
      write(*,*)
   endif

   call read_value_string_p(OUTPUT_FILES, 'OUTPUT_FILES')
   if (err_occurred() /= 0) then
      some_parameters_missing_from_Par_file = .true.
      write(*,'(a)') 'OUTPUT_FILES                    = ./OUTPUT_FILES'
      write(*,*)
   endif
   call create_directory_if_doesnt_exist(OUTPUT_FILES)

   ! deprecated: call read_value_integer_p(partitioning_method, 'partitioning_method')
   call read_value_integer_p(PARTITIONING_TYPE, 'PARTITIONING_TYPE')
   if (err_occurred() /= 0) then
      ! old version
      call read_value_integer_p(PARTITIONING_TYPE, 'partitioning_method')
      if (err_occurred() == 0) then
         ! deprecation warning
         write(*,'(a)') 'Warning: Deprecated parameter partitioning_method found in Par_file.'
         write(*,'(a)') '         Please use parameter PARTITIONING_TYPE in future...'
      else
         some_parameters_missing_from_Par_file = .true.
         write(*,'(a)') 'PARTITIONING_TYPE               = 3'
         write(*,*)
      endif
   endif

   ! deprecated: call read_value_integer_p(ngnod, 'ngnod')
   call read_value_integer_p(NGNOD, 'NGNOD')
   if (err_occurred() /= 0) then
      ! old version
      call read_value_integer_p(NGNOD, 'ngnod')
      if (err_occurred() == 0) then
         ! deprecation warning
         write(*,'(a)') 'Warning: Deprecated parameter ngnod found in Par_file.'
         write(*,'(a)') '         Please use parameter NGNOD in future...'
      else
         some_parameters_missing_from_Par_file = .true.
         write(*,'(a)') 'NGNOD                           = 9'
         write(*,*)
      endif
   endif

   call read_value_string_p(database_filename, 'database_filename')
   if (err_occurred() /= 0) then
      some_parameters_missing_from_Par_file = .true.
      write(*,'(a)') 'database_filename               = ./DATA/database'
      write(*,*)
   endif

   call read_value_logical_p(STACEY_ABSORBING_CONDITIONS, 'STACEY_ABSORBING_CONDITIONS')
   if (err_occurred() /= 0) then
      some_parameters_missing_from_Par_file = .true.
      write(*,'(a)') 'STACEY_ABSORBING_CONDITIONS     = .true.'
      write(*,*)
   endif

   !--------------------------------------------------------------------
   !
   ! receivers
   !
   !--------------------------------------------------------------------

   call read_value_logical_p(use_existing_STATIONS, 'use_existing_STATIONS')
   if (err_occurred() /= 0) then
      some_parameters_missing_from_Par_file = .true.
      write(*,'(a)') 'use_existing_STATIONS           = .false.'
      write(*,*)
   endif

   call read_value_integer_p(nreceiversets, 'nreceiversets')
   if (err_occurred() /= 0) then
      some_parameters_missing_from_Par_file = .true.
      write(*,'(a)') 'nreceiversets                   = 1'
      write(*,*)
   endif

   call read_value_double_precision_p(anglerec, 'anglerec')
   if (err_occurred() /= 0) then
      some_parameters_missing_from_Par_file = .true.
      write(*,'(a)') 'anglerec                        = 0.d0'
      write(*,*)
   endif

   call read_value_logical_p(rec_normal_to_surface, 'rec_normal_to_surface')
   if (err_occurred() /= 0) then
      some_parameters_missing_from_Par_file = .true.
      write(*,'(a)') 'rec_normal_to_surface           = .false.'
      write(*,*)
   endif

   call read_value_string_p(stations_filename, 'stations_filename')
   if (err_occurred() /= 0) then
      some_parameters_missing_from_Par_file = .true.
      write(*,'(a)') 'stations_filename               = ./DATA/STATIONS'
      write(*,*)
   endif

   ! receiver sets will be read in later...

   !--------------------------------------------------------------------
   !
   ! velocity and density models
   !
   !--------------------------------------------------------------------

   ! read the different material materials (i.e. the number of models)
   call read_value_integer_p(nbmodels, 'nbmodels')
   if (err_occurred() /= 0) then
      some_parameters_missing_from_Par_file = .true.
      write(*,'(a)') 'nbmodels                        = 1'
      write(*,*)
   endif

   ! material definitions will be read later on...

   call read_value_string_p(TOMOGRAPHY_FILE, 'TOMOGRAPHY_FILE')
   if (err_occurred() /= 0) then
      some_parameters_missing_from_Par_file = .true.
      write(*,'(a)') 'TOMOGRAPHY_FILE                 = ./DATA/tomo_file.xyz'
      write(*,*)
   endif

   ! boolean defining whether internal or external mesh
   call read_value_logical_p(read_external_mesh, 'read_external_mesh')
   if (err_occurred() /= 0) then
      some_parameters_missing_from_Par_file = .true.
      write(*,'(a)') 'read_external_mesh              = .false.'
      write(*,*)
   endif

   !--------------------------------------------------------------------
   !
   ! parameters external / internal meshing
   !
   !--------------------------------------------------------------------

   !-----------------
   ! external mesh parameters

   if (read_external_mesh) then

      ! read info about external mesh
      call read_value_string_p(mesh_file, 'mesh_file')
      if (err_occurred() /= 0) then
         some_parameters_missing_from_Par_file = .true.
         write(*,'(a)') 'mesh_file                       = ./DATA/mesh_file'
         write(*,*)
      endif

      call read_value_string_p(nodes_coords_file, 'nodes_coords_file')
      if (err_occurred() /= 0) then
         some_parameters_missing_from_Par_file = .true.
         write(*,'(a)') 'nodes_coords_file               = ./DATA/nodes_coords_file'
         write(*,*)
      endif

      call read_value_string_p(materials_file, 'materials_file')
      if (err_occurred() /= 0) then
         some_parameters_missing_from_Par_file = .true.
         write(*,'(a)') 'materials_file                  = ./DATA/materials_file'
         write(*,*)
      endif

      call read_value_string_p(free_surface_file, 'free_surface_file')
      if (err_occurred() /= 0) then
         some_parameters_missing_from_Par_file = .true.
         write(*,'(a)') 'free_surface_file               = ./DATA/free_surface_file'
         write(*,*)
      endif

      call read_value_string_p(axial_elements_file, 'axial_elements_file')
      if (err_occurred() /= 0) then
         some_parameters_missing_from_Par_file = .true.
         write(*,'(a)') 'axial_elements_file             = ./DATA/axial_elements_file'
         write(*,*)
      endif

      call read_value_string_p(absorbing_surface_file, 'absorbing_surface_file')
      if (err_occurred() /= 0) then
         some_parameters_missing_from_Par_file = .true.
         write(*,'(a)') 'absorbing_surface_file          = ./DATA/absorbing_surface_file'
         write(*,*)
      endif

      call read_value_string_p(acoustic_forcing_surface_file, 'acoustic_forcing_surface_file')
      if (err_occurred() /= 0) then
         some_parameters_missing_from_Par_file = .true.
         write(*,'(a)') 'acoustic_forcing_surface_file   = ./DATA/MSH/Surf_acforcing_Bottom_enforcing_mesh'
         write(*,*)
      endif

      call read_value_string_p(absorbing_cpml_file, 'absorbing_cpml_file')
      if (err_occurred() /= 0) then
         some_parameters_missing_from_Par_file = .true.
         write(*,'(a)') 'absorbing_cpml_file             = ./DATA/absorbing_cpml_file'
         write(*,*)
      endif

      call read_value_string_p(tangential_detection_curve_file, 'tangential_detection_curve_file')
      if (err_occurred() /= 0) then
         some_parameters_missing_from_Par_file = .true.
         write(*,'(a)') 'tangential_detection_curve_file = ./DATA/courbe_eros_nodes'
         write(*,*)
      endif

      call read_value_string_p(nonconforming_adjacencies_file, 'nonconforming_adjacencies_file')
      if (err_occurred() /= 0) then
         should_read_nonconforming_adjacencies_file = .false.
      else
         should_read_nonconforming_adjacencies_file = .true.
      endif

   else

      !-----------------
      ! internal mesh parameters

      ! interfaces file
      call read_value_string_p(interfaces_filename, 'interfacesfile')
      if (err_occurred() /= 0) then
         some_parameters_missing_from_Par_file = .true.
         write(*,'(a)') 'interfaces_file                  = DATA/interfaces_simple_topo_curved.dat'
         write(*,*)
      endif

      ! read grid parameters
      call read_value_double_precision_p(xmin_param, 'xmin')
      if (err_occurred() /= 0) then
         some_parameters_missing_from_Par_file = .true.
         write(*,'(a)') 'xmin                            = 0.d0'
         write(*,*)
      endif

      call read_value_double_precision_p(xmax_param, 'xmax')
      if (err_occurred() /= 0) then
         some_parameters_missing_from_Par_file = .true.
         write(*,'(a)') 'xmax                            = 4000.d0'
         write(*,*)
      endif

      call read_value_integer_p(nx_param, 'nx')
      if (err_occurred() /= 0) then
         some_parameters_missing_from_Par_file = .true.
         write(*,'(a)') 'nx                              = 80'
         write(*,*)
      endif

      ! read absorbing boundary parameters
      call read_value_logical_p(absorbbottom, 'absorbbottom')
      if (err_occurred() /= 0) then
         some_parameters_missing_from_Par_file = .true.
         write(*,'(a)') 'absorbbottom                    = .true.'
         write(*,*)
      endif

      call read_value_logical_p(absorbright, 'absorbright')
      if (err_occurred() /= 0) then
         some_parameters_missing_from_Par_file = .true.
         write(*,'(a)') 'absorbright                     = .true.'
         write(*,*)
      endif

      call read_value_logical_p(absorbtop, 'absorbtop')
      if (err_occurred() /= 0) then
         some_parameters_missing_from_Par_file = .true.
         write(*,'(a)') 'absorbtop                       = .false.'
         write(*,*)
      endif

      call read_value_logical_p(absorbleft, 'absorbleft')
      if (err_occurred() /= 0) then
         some_parameters_missing_from_Par_file = .true.
         write(*,'(a)') 'absorbleft                      = .true.'
         write(*,*)
      endif

      call read_value_integer_p(nbregions, 'nbregions')
      if (err_occurred() /= 0) then
         some_parameters_missing_from_Par_file = .true.
         write(*,'(a)') 'nbregions                       = 1'
         write(*,*)
      endif
! note: if internal mesh, then region tables will be read in by read_regions (from meshfem2D)
   endif

   !--------------------------------------------------------------------
   ! Database output parameters
   !--------------------------------------------------------------------

   call read_value_logical_p(write_adjacency_map, 'write_adjacency_map')
   if (err_occurred() /= 0) then
      write(*,*) 'Warning: write_adjacency_map parameter not found in Par_file, setting to .true. by default.'
      write_adjacency_map = .true.
   endif

   !--------------------------------------------------------------------
   ! Display parameters
   !--------------------------------------------------------------------

   call read_value_logical_p(output_grid_Gnuplot, 'output_grid_Gnuplot')
   call read_value_logical_p(output_grid_ASCII, 'output_grid_ASCII')



   if (some_parameters_missing_from_Par_file) then
      write(*,*)
      write(*,*) 'All the above parameters are missing from your Par_file.'
      write(*,*) 'Please cut and paste them somewhere in your Par_file (any place is fine), change their values if needed'
      write(*,*) '(the above values are just default values), and restart your run.'
      write(*,*)
      call stop_the_code('Error: some parameters are missing in your Par_file, it is incomplete or in an older format, &
      &see at the end of the standard output file of the run for detailed and easy instructions about how to fix that')
   endif

   !--------------------------------------------------------------------

   ! converts all string characters to lowercase
   irange = iachar('a') - iachar('A')
   do i = 1,len_trim(MODEL)
      if (lge(MODEL(i:i),'A') .and. lle(MODEL(i:i),'Z')) then
         MODEL(i:i) = achar(iachar(MODEL(i:i)) + irange)
      endif
   enddo
   do i = 1,len_trim(SAVE_MODEL)
      if (lge(SAVE_MODEL(i:i),'A') .and. lle(SAVE_MODEL(i:i),'Z')) then
         SAVE_MODEL(i:i) = achar(iachar(SAVE_MODEL(i:i)) + irange)
      endif
   enddo

   ! checks input parameters
   call check_parameters()

end subroutine read_parameter_file_only

!
!-------------------------------------------------------------------------------------------------
!

subroutine check_parameters()

   use shared_parameters

   implicit none

   ! checks partitioning
   if (NPROC <= 0) then
      print *, 'Error: Number of processes (NPROC) must be greater than or equal to one.'
      call stop_the_code('Error invalid NPROC value')
   endif

#ifndef WITH_MPI
   if (NPROC > 1) then
      print *, 'Error: Number of processes (NPROC) must be equal to one when not using MPI.'
      print *, 'Please recompile with -DWITH_MPI in order to enable use of MPI.'
      call stop_the_code('Error invalid NPROC value')
   endif
#endif

   if (PARTITIONING_TYPE /= 1 .and. PARTITIONING_TYPE /= 3) then
      print *, 'Error: Invalid partitioning method number.'
      print *, 'Partitioning type ',PARTITIONING_TYPE,' was requested, but is not available.'
      print *, 'Support for the METIS graph partitioner has been discontinued, please use SCOTCH (option 3) instead.'
      call stop_the_code('Error invalid partitioning method')
   endif

   ! if ( NGNOD /= 9) &
   !    call stop_the_code('NGNOD should be 9!')

   ! reads in material definitions
   if (nbmodels <= 0) &
      call stop_the_code('Non-positive number of materials not allowed!')

   ! check regions
   if (read_external_mesh .eqv. .false.) then
      if (nbregions <= 0) call stop_the_code('Negative number of regions not allowed for internal meshing!')
   endif

end subroutine check_parameters

!
!-------------------------------------------------------------------------------------------------
!

subroutine read_parameter_file_receiversets()

   use constants, only: IMAIN,IIN,IN_DATA_FILES,mygroup

   use shared_parameters

   implicit none

   ! local parameters
   integer :: ireceiverlines,ier,nrec
   logical :: reread_rec_normal_to_surface
   character(len=MAX_STRING_LEN) :: path_to_add,dummystring

   integer,external :: err_occurred

   ! re-reads rec_normal_to_surface parameter to reposition read header for following next-line reads
   call read_value_logical_p(reread_rec_normal_to_surface, 'rec_normal_to_surface')
   if (err_occurred() /= 0) call stop_the_code('error reading parameter rec_normal_to_surface in Par_file')

   ! checks
   if (reread_rec_normal_to_surface .neqv. rec_normal_to_surface) &
      call stop_the_code('Invalid re-reading of rec_normal_to_surface parameter')

   ! reads in receiver sets
   if (use_existing_STATIONS) then
      ! checks if STATIONS file exisits

      ! adds specific run folder to path
      ! for example: run0001/DATA/STATIONS
      if (NUMBER_OF_SIMULTANEOUS_RUNS > 1 .and. mygroup >= 0) then
         write(path_to_add,"('run',i4.4,'/')") mygroup + 1
         stations_filename = path_to_add(1:len_trim(path_to_add))//stations_filename(1:len_trim(stations_filename))
      endif

      ! user output
      write(IMAIN,*) '  using existing STATIONS file: ',trim(stations_filename)
      call flush_IMAIN()

      ! counts entries
      open(unit=IIN,file=trim(stations_filename),status='old',action='read',iostat=ier)
      if (ier /= 0 ) then
         print *, 'Error could not open existing STATIONS file:',trim(stations_filename)
         print *, 'Please check if file exists.'
         call stop_the_code('Error opening STATIONS file')
      endif

      nrec = 0
      do while(ier == 0)
         read(IIN,"(a)",iostat=ier) dummystring
         if (ier == 0) then
            ! skip empty lines
            if (len_trim(dummystring) == 0) cycle

            ! skip comment lines
            dummystring = adjustl(dummystring)
            if (dummystring(1:1) == "#") cycle

            ! increase counter
            nrec = nrec + 1
         endif
      enddo
      close(IIN)

      write(IMAIN,*) '  file name is ',trim(stations_filename)
      write(IMAIN,*) '  found ',nrec,' receivers'
      write(IMAIN,*)

   else
      ! receiver lines specified in Par_file
      ! only valid if at least 1 receiver line is specified
      if (nreceiversets < 1) &
         call stop_the_code('number of receiver sets must be greater than 1')

      ! allocate receiver line arrays
      allocate(nrec_line(nreceiversets))
      allocate(xdeb(nreceiversets))
      allocate(zdeb(nreceiversets))
      allocate(xfin(nreceiversets))
      allocate(zfin(nreceiversets))
      allocate(record_at_surface_same_vertical(nreceiversets),stat=ier)
      if (ier /= 0 ) call stop_the_code('Error allocating receiver lines')

      nrec_line(:) = 0
      xdeb(:) = 0.d0
      zdeb(:) = 0.d0
      xfin(:) = 0.d0
      zfin(:) = 0.d0
      record_at_surface_same_vertical(:) = .false.

      ! loop on all the receiver lines
      do ireceiverlines = 1,nreceiversets
         call read_value_integer_next_p(nrec_line(ireceiverlines),'nrec')
         if (err_occurred() /= 0) call stop_the_code('error reading parameter nrec in Par_file')

         call read_value_double_prec_next_p(xdeb(ireceiverlines),'xdeb')
         if (err_occurred() /= 0) call stop_the_code('error reading parameter xdeb in Par_file')

         call read_value_double_prec_next_p(zdeb(ireceiverlines),'zdeb')
         if (err_occurred() /= 0) call stop_the_code('error reading parameter zdeb in Par_file')

         call read_value_double_prec_next_p(xfin(ireceiverlines),'xfin')
         if (err_occurred() /= 0) call stop_the_code('error reading parameter xfin in Par_file')

         call read_value_double_prec_next_p(zfin(ireceiverlines),'zfin')
         if (err_occurred() /= 0) call stop_the_code('error reading parameter zfin in Par_file')

         call read_value_logical_next_p(record_at_surface_same_vertical(ireceiverlines),'record_at_surface_same_vertical')
         if (err_occurred() /= 0) call stop_the_code('error reading parameter record_at_surface_same_vertical in Par_file')

         if (read_external_mesh .and. record_at_surface_same_vertical(ireceiverlines)) then
            call stop_the_code('Cannot use record_at_surface_same_vertical with external meshes!')
         endif
      enddo
   endif

end subroutine read_parameter_file_receiversets

!
!-------------------------------------------------------------------------------------------------
!

subroutine read_parameter_file_derive_flags()

   use shared_parameters

   implicit none

   ! derives additional flags based on input parameters

   ! sets overal Bielak flag
   add_Bielak_conditions = add_Bielak_conditions_bottom .or. add_Bielak_conditions_right .or. &
      add_Bielak_conditions_top .or. add_Bielak_conditions_left

   ! boundary conditions
   if (add_Bielak_conditions .and. .not. STACEY_ABSORBING_CONDITIONS) &
      call stop_the_code('need STACEY_ABSORBING_CONDITIONS set to .true. in order to use add_Bielak_conditions')

   ! solve the conflict in value of PML_BOUNDARY_CONDITIONS and STACEY_ABSORBING_CONDITIONS
   if (PML_BOUNDARY_CONDITIONS) any_abs = .true.
   if (STACEY_ABSORBING_CONDITIONS) any_abs = .true.

   ! initializes flags for absorbing boundaries
   if (.not. any_abs) then
      absorbbottom = .false.
      absorbright = .false.
      absorbtop = .false.
      absorbleft = .false.
   endif

   ! can use only one point to display lower-left corner only for interpolated snapshot
   if (pointsdisp < 3) then
      pointsdisp = 3
      plot_lowerleft_corner_only = .true.
   else
      plot_lowerleft_corner_only = .false.
   endif

end subroutine read_parameter_file_derive_flags

!
!-------------------------------------------------------------------------------------------------
!

subroutine open_parameter_file_from_main_only()

   use constants, only: MAX_STRING_LEN,IN_DATA_FILES

   implicit none

   character(len=MAX_STRING_LEN) :: filename_main,filename_run0001
   logical :: exists_main_Par_file,exists_run0001_Par_file
   integer :: ier

   filename_main = IN_DATA_FILES(1:len_trim(IN_DATA_FILES))//'Par_file'

! also see if we are running several independent runs in parallel
! to do so, add the right directory for that run for the main process only here
   filename_run0001 = 'run0001/'//filename_main(1:len_trim(filename_main))
   call param_open(filename_main, len(filename_main), ier)
   if (ier == 0) then
      exists_main_Par_file = .true.
      call close_parameter_file()
   else
      exists_main_Par_file    = .false.
   endif
   call param_open(filename_run0001, len(filename_run0001), ier)
   if (ier == 0) then
      exists_run0001_Par_file = .true.
      call close_parameter_file()
   else
      exists_run0001_Par_file = .false.
   endif

   !if (exists_main_Par_file .and. exists_run0001_Par_file) then ! TODO why is it like that in the 3D version??
   !  print *
   !  print *,'cannot have both DATA/Par_file and run0001/DATA/Par_file present, please remove one of them'
   !  stop 'error: two different copies of the Par_file'
   !endif

   call param_open(filename_main, len(filename_main), ier)
   if (ier /= 0) then
      call param_open(filename_run0001, len(filename_run0001), ier)
      if (ier /= 0) then
         print *
         print *,'opening file failed, please check your file path and run-directory.'
         call stop_the_code('error opening Par_file')
      endif
   endif

end subroutine open_parameter_file_from_main_only

!
!-------------------------------------------------------------------------------------------------
!

subroutine open_parameter_file()

   use constants, only: MAX_STRING_LEN,mygroup,IN_DATA_FILES
   use shared_parameters, only: NUMBER_OF_SIMULTANEOUS_RUNS, Par_File

   implicit none

   integer ierr
   common /param_err_common/ ierr
   character(len=MAX_STRING_LEN) filename,path_to_add

   ! Par_file filename with path
   filename = Par_File

   ! see if we are running several independent runs in parallel
   ! if so, add the right directory for that run
   ! (group numbers start at zero, but directory names start at run0001, thus we add one)
   !
   ! a negative value for "mygroup" is a convention that indicates that groups (i.e. sub-communicators, one per run) are off
   if (NUMBER_OF_SIMULTANEOUS_RUNS > 1 .and. mygroup >= 0) then
      write(path_to_add,"('run',i4.4,'/')") mygroup + 1
      filename = path_to_add(1:len_trim(path_to_add))//filename(1:len_trim(filename))
   endif

   ! to use c routines for reading/parsing file content
   call param_open(filename, len_trim(filename), ierr)
   if (ierr /= 0) then
      print *
      print *,'opening file failed, please check your file path and run-directory.'
      call stop_the_code('error opening Par_file')
   endif

end subroutine open_parameter_file

!
!-------------------------------------------------------------------------------------------------
!

subroutine close_parameter_file()

   implicit none

   ! to use C routines
   call param_close()

end subroutine close_parameter_file
