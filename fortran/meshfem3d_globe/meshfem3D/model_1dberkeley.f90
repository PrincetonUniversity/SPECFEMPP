!=====================================================================
!
!                       S p e c f e m 3 D  G l o b e
!                       ----------------------------
!
!     Main historical authors: Dimitri Komatitsch and Jeroen Tromp
!                        Princeton University, USA
!                and CNRS / University of Marseille, France
!                 (there are currently many more authors!)
! (c) Princeton University and CNRS / University of Marseille, April 2014
!
! This program is free software; you can redistribute it and/or modify
! it under the terms of the GNU General Public License as published by
! the Free Software Foundation; either version 3 of the License, or
! (at your option) any later version.
!
! This program is distributed in the hope that it will be useful,
! but WITHOUT ANY WARRANTY; without even the implied warranty of
! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
! GNU General Public License for more details.
!
! You should have received a copy of the GNU General Public License along
! with this program; if not, write to the Free Software Foundation, Inc.,
! 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
!
!=====================================================================

!-------------------------------
!
! 1D Berkeley model
!
! 1D reference model for SEMUCB - WM1
! radially anisotropic shear-wave model
!
!-------------------------------

module model_1dberkeley_par

  ! Added by < FM> Feb. 2022
  use constants, only: A3d_folder

  implicit none

  ! number of layers in model1Dberkeley.dat
  integer :: NR_REF_BERKELEY
  integer :: NR_inner_core_berk
  integer :: NR_outer_core_berk
  integer :: NR_water_berk
  integer :: ifanis_berk
  integer :: tref_berk
  integer :: ifdeck_berk

  ! model_1dberkeley_variables
  double precision, dimension(:), allocatable :: &
    Mref_V_radius_berkeley, &
    Mref_V_density_berkeley, &
    Mref_V_vpv_berkeley, &
    Mref_V_vph_berkeley, &
    Mref_V_vsv_berkeley, &
    Mref_V_vsh_berkeley, &
    Mref_V_eta_berkeley, &
    Mref_V_Qkappa_berkeley, &
    Mref_V_Qmu_berkeley

  ! Berkeley 1D model
  character(len=*), parameter :: berkeley_file_model1D = trim(A3d_folder) // 'model1D.dat'

  ! moho layer index
  integer :: moho1D_layer_index = -1
  ! moho radius from 1D reference model (in km)
  double precision :: moho1D_radius = 0.0d0
  ! moho depth from 1D reference model (in km)
  double precision :: moho1D_depth = 0.0d0

end module model_1dberkeley_par

!
!--------------------------------------------------------------------------------------------------
!

  subroutine model_1dberkeley_broadcast()

! reads and broadcasts berkeley 1D model

  use model_1dberkeley_par
  use constants, only: myrank

  implicit none
  ! local parameters
  integer :: ier

  ! define the berkeley 1D model
  ! Utpal Kumar, Feb, 2022

  ! main process reads in model
  if (myrank == 0) call read_1dberkeley()

  ! broadcast header values
  call bcast_all_singlei(ifanis_berk       )    ! ifanis_berk is not really needed any further, but just in case
  call bcast_all_singlei(tref_berk         )    ! tref_berk is not really needed any further, but just in case
  call bcast_all_singlei(ifdeck_berk       )    ! ifdeck_berk is not really needed any further, but just in case
  call bcast_all_singlei(NR_REF_BERKELEY   )
  call bcast_all_singlei(NR_inner_core_berk)
  call bcast_all_singlei(NR_outer_core_berk)
  call bcast_all_singlei(NR_water_berk     )
  call bcast_all_singlei(moho1D_layer_index)    ! moho info
  call bcast_all_singledp(moho1D_radius)        ! radius & depth only needed on main process, but just in case
  call bcast_all_singledp(moho1D_depth)

  ! allocate arrays
  if (.not. allocated(Mref_V_radius_berkeley)) then
    allocate(Mref_V_radius_berkeley(NR_REF_BERKELEY), &
             Mref_V_density_berkeley(NR_REF_BERKELEY), &
             Mref_V_vpv_berkeley(NR_REF_BERKELEY), &
             Mref_V_vsv_berkeley(NR_REF_BERKELEY), &
             Mref_V_Qkappa_berkeley(NR_REF_BERKELEY), &
             Mref_V_Qmu_berkeley(NR_REF_BERKELEY), &
             Mref_V_vph_berkeley(NR_REF_BERKELEY), &
             Mref_V_vsh_berkeley(NR_REF_BERKELEY), &
             Mref_V_eta_berkeley(NR_REF_BERKELEY),stat=ier)
    if (ier /= 0) stop 'Error allocating 1D Berkeley model arrays'
  endif

  ! broadcast data
  call BCAST_ALL_DP(Mref_V_radius_berkeley ,NR_REF_BERKELEY)
  call BCAST_ALL_DP(Mref_V_density_berkeley,NR_REF_BERKELEY)
  call BCAST_ALL_DP(Mref_V_vpv_berkeley    ,NR_REF_BERKELEY)
  call BCAST_ALL_DP(Mref_V_vph_berkeley    ,NR_REF_BERKELEY)
  call BCAST_ALL_DP(Mref_V_vsv_berkeley    ,NR_REF_BERKELEY)
  call BCAST_ALL_DP(Mref_V_vsh_berkeley    ,NR_REF_BERKELEY)
  call BCAST_ALL_DP(Mref_V_eta_berkeley    ,NR_REF_BERKELEY)
  call BCAST_ALL_DP(Mref_V_Qkappa_berkeley ,NR_REF_BERKELEY)
  call BCAST_ALL_DP(Mref_V_Qmu_berkeley    ,NR_REF_BERKELEY)

  end subroutine model_1dberkeley_broadcast

!
!--------------------------------------------------------------------------------------------------
!

  subroutine read_1dberkeley()

  use model_1dberkeley_par
  use constants, only: myrank,IMAIN,IIN,TINYVAL,ATTENUATION_1D_WITH_3D_STORAGE

  implicit none
  ! local parameters
  integer :: i,ier
  character (len=100) :: title
  double precision :: r,rho,vpv,vph,vsv,vsh,eta,qmu,qkappa
  double precision :: ifanis,tref,ifdeck

  ! user output
  if (myrank == 0) then
    write(IMAIN,*) 'reference model: 1D Berkeley'
    write(IMAIN,*) '  model file             = ',trim(berkeley_file_model1D)
    write(IMAIN,*)
    call flush_IMAIN()
  endif

  ! safety check
  if (.not. ATTENUATION_1D_WITH_3D_STORAGE) then
    ! note: The 1D Berkeley model as given in the SEMUCB_A3d/model1D.dat file uses 713 layers from core to top surface.
    !       Attenuation parameters Qmu (and Qkappa) are interpolated with depth between layers.
    !       This leads to smooth variations within elements and thus requires 3D storage for attenuation arrays.
    !
    !       We add this check to make sure the code is compiled with the correct setting to produce seismograms
    !       for the most accurate Berkeley model possible.
    print *,'Error: Invalid ATTENUATION_1D_WITH_3D_STORAGE == .false. flag for 1D Berkeley reference model!'
    print *
    print *,'Using the 1D Berkeley reference model needs attenuation parameters stored as 3D arrays.'
    print *,'Please change the flag ATTENUATION_1D_WITH_3D_STORAGE to .true. in setup/constants.h, and re-compile the code.'
    print *
    ! stop
    stop 'Invalid ATTENUATION_1D_WITH_3D_STORAGE flag for 1D Berkeley reference model'
  endif

  ! initializes number of layers
  NR_REF_BERKELEY = 0
  NR_inner_core_berk = 0
  NR_outer_core_berk = 0
  NR_water_berk = 0

  ! only main process reads in file
  open(IIN,file=trim(berkeley_file_model1D),status='old',iostat=ier)
  if (ier /= 0) then
    print *,'Error opening file: ',trim(berkeley_file_model1D)
    stop 'Error opening file model1D.dat for Berkeley reference model'
  endif

  read(IIN,100,iostat=ier) title
  if (ier /= 0) stop 'Error reading title in model1D.dat for Berkeley reference model'

  read(IIN,*  ,iostat=ier) ifanis, tref, ifdeck
  if (ier /= 0) stop 'Error reading ifanis/tref/ifdeck in model1D.dat for Berkeley reference model'

  ! stores all as integers - not needed any further yet, but just in case...
  ifanis_berk = int(ifanis)
  tref_berk = int(tref)
  ifdeck_berk = int(ifdeck)

  read(IIN,*  ,iostat=ier) NR_REF_BERKELEY, NR_inner_core_berk, NR_outer_core_berk, NR_water_berk
  if (ier /= 0) stop 'Error reading NR_REF/NR_inner_core/NR_outer_core/NR_water in model1D.dat for Berkeley reference model'

  ! user output
  if (myrank == 0) then
    write(IMAIN,*) '  title                  = ',trim(title)
    write(IMAIN,*) '  total number of layers = ',NR_REF_BERKELEY
    write(IMAIN,*)
    call flush_IMAIN()
  endif

  ! allocate arrays (only on main process)
  allocate(Mref_V_radius_berkeley(NR_REF_BERKELEY), &
           Mref_V_density_berkeley(NR_REF_BERKELEY), &
           Mref_V_vpv_berkeley(NR_REF_BERKELEY), &
           Mref_V_vsv_berkeley(NR_REF_BERKELEY), &
           Mref_V_Qkappa_berkeley(NR_REF_BERKELEY), &
           Mref_V_Qmu_berkeley(NR_REF_BERKELEY), &
           Mref_V_vph_berkeley(NR_REF_BERKELEY), &
           Mref_V_vsh_berkeley(NR_REF_BERKELEY), &
           Mref_V_eta_berkeley(NR_REF_BERKELEY),stat=ier)
  if (ier /= 0) stop 'Error allocating 1D Berkeley model arrays'
  Mref_V_radius_berkeley(:) = 0.d0
  Mref_V_density_berkeley(:) = 0.d0
  Mref_V_vpv_berkeley(:) = 0.d0
  Mref_V_vsv_berkeley(:) = 0.d0
  Mref_V_Qkappa_berkeley(:) = 0.d0
  Mref_V_Qmu_berkeley(:) = 0.d0
  Mref_V_vph_berkeley(:) = 0.d0
  Mref_V_vsh_berkeley(:) = 0.d0
  Mref_V_eta_berkeley(:) = 0.d0

  ! reads data
  do i = 1,NR_REF_BERKELEY
    ! format: #radius #density #vpv #vsv #Qkappa #Qmu #vph #vsh #eta
    read(IIN,*,iostat=ier) r,rho,vpv,vsv,qkappa,qmu,vph,vsh,eta
    if (ier /= 0) then
      print *,'Error: reading layer',i,' for reference Berkeley model'
      print *,'Please check model format with routine read_1dberkeley() in file model_1dberkeley.f90 ...'
      stop 'Error reading layer in model1D.dat for Berkeley reference model'
    endif

    ! stores layer values
    Mref_V_radius_berkeley(i) = r
    Mref_V_density_berkeley(i) = rho
    Mref_V_vpv_berkeley(i) = vpv
    Mref_V_vsv_berkeley(i) = vsv
    Mref_V_Qkappa_berkeley(i) = qkappa
    Mref_V_Qmu_berkeley(i) = qmu
    Mref_V_vph_berkeley(i) = vph
    Mref_V_vsh_berkeley(i) = vsh
    Mref_V_eta_berkeley(i) = eta
  enddo

  close(IIN)

  ! checks model consistency between number of layers and values given
  do i = 1,NR_REF_BERKELEY
    ! shear velocities in layer for checks
    vsv = Mref_V_vsv_berkeley(i)
    vsh = Mref_V_vsh_berkeley(i)

    ! checks layer
    if (i <= NR_inner_core_berk) then
      ! inner core layers (solid)
      ! must have non-zero vsv and vsh
      ! Vsv
      if (vsv < TINYVAL) then
        print *,'Error: invalid Vsv velocity ',vsv,' (m/s) in inner core layer',i,' for reference Berkeley model'
        stop 'Invalid Vsv value in inner core'
      endif
      ! Vsh
      if (vsh < TINYVAL) then
        print *,'Error: invalid Vsh velocity ',vsh,' (m/s) in inner core layer',i,' for reference Berkeley model'
        stop 'Invalid Vsh value in inner core'
      endif
    else if (i <= NR_outer_core_berk) then
      ! outer core layers (fluid)
      ! must have zero vsv and vsh
      ! Vsv
      if (abs(vsv) > TINYVAL) then
        print *,'Error: invalid Vsv velocity ',vsv,' (m/s) in outer core layer',i,' for reference Berkeley model'
        stop 'Invalid Vsv value in outer core'
      endif
      ! Vsh
      if (abs(vsh) > TINYVAL) then
        print *,'Error: invalid Vsh velocity ',vsh,' (m/s) in outer core layer',i,' for reference Berkeley model'
        stop 'Invalid Vsh value in outer core'
      endif
    else if (i <= NR_REF_BERKELEY - NR_water_berk) then
      ! crust/mantle layers (solid)
      ! must have non-zero vsv and vsh
      ! Vsv
      if (vsv < TINYVAL) then
        print *,'Error: invalid Vsv velocity ',vsv,' (m/s) in crust/mantle layer',i,' for reference Berkeley model'
        stop 'Invalid Vsv value in crust/mantle'
      endif
      ! Vsh
      if (vsh < TINYVAL) then
        print *,'Error: invalid Vsh velocity ',vsh,' (m/s) in crust/mantle layer',i,' for reference Berkeley model'
        stop 'Invalid Vsh value in crust/mantle'
      endif
    else
      ! water/ocean layers (fluid)
      ! must have zero vsv and vsh
      ! Vsv
      if (abs(vsv) > TINYVAL) then
        print *,'Error: invalid Vsv velocity ',vsv,' (m/s) in ocean layer',i,' for reference Berkeley model'
        stop 'Invalid Vsv value in ocean layer'
      endif
      ! Vsh
      if (abs(vsh) > TINYVAL) then
        print *,'Error: invalid Vsh velocity ',vsh,' (m/s) in ocean layer',i,' for reference Berkeley model'
        stop 'Invalid Vsh value in ocean layer'
      endif
    endif
  enddo

  ! determine node index where crust starts
  call determine_1dberkeley_moho_layer()

  !
  ! reading formats
  !
100 format(a80)
!105 format(f8.0, 3f9.2, 2f9.1, 2f9.2, f9.5)

  end subroutine read_1dberkeley

!
!--------------------------------------------------------------------------------------------------
!

  subroutine model_1dberkeley(x,rho,vpv,vph,vsv,vsh,eta,Qkappa,Qmu,iregion_code,CRUSTAL)

  use model_1dberkeley_par
  use constants, only: PI,GRAV,EARTH_RHOAV,EARTH_R,EARTH_R_KM, &
    IREGION_INNER_CORE,IREGION_OUTER_CORE,IREGION_CRUST_MANTLE

  implicit none

! model_1dref_variables

! input:
! dimensionless radius x

! output: non-dimensionalized
!
! mass density             : rho
! compressional wave speed : vpv
! compressional wave speed : vph
! shear wave speed         : vsv
! shear wave speed         : vsh
! dimensionless parameter  : eta
! shear quality factor     : Qmu
! bulk quality factor      : Qkappa

  double precision,intent(in) :: x
  double precision,intent(inout) :: rho,vpv,vph,vsv,vsh,eta,Qmu,Qkappa
  integer,intent(in) :: iregion_code
  logical,intent(in) :: CRUSTAL

  ! local parameters
  double precision :: r,frac,scaleval
  integer :: i
  logical, parameter :: MIMIC_NATIVE_SPECFEM = .true.

  ! compute real physical radius in meters
  r = x * EARTH_R

  i = 1
  do while(r >= Mref_V_radius_berkeley(i) .and. i /= NR_REF_BERKELEY)
    i = i + 1
  enddo

  ! make sure we stay in the right region
  if (MIMIC_NATIVE_SPECFEM) then
    ! inner core bounds
    if (iregion_code == IREGION_INNER_CORE) then
      if (i > NR_inner_core_berk) i = NR_inner_core_berk
    endif
    ! outer core bounds
    if (iregion_code == IREGION_OUTER_CORE) then
      if (i < NR_inner_core_berk+2) i = NR_inner_core_berk+2
      if (i > NR_outer_core_berk) i = NR_outer_core_berk
    endif
    ! crust/mantle bounds
    if (iregion_code == IREGION_CRUST_MANTLE) then
      if (i < NR_outer_core_berk+2) i = NR_outer_core_berk+2
    endif

    ! if crustal model is used, mantle gets expanded up to surface
    ! for any depth less than 24.4 km, values from mantle below moho are taken
    if (CRUSTAL) then
      if (i > moho1D_layer_index) i = moho1D_layer_index ! Warning : may need to be changed if file is modified !
    endif
  endif

  if (i == 1) then
    ! first layer in inner core
    rho    = Mref_V_density_berkeley(i)
    vpv    = Mref_V_vpv_berkeley(i)
    vph    = Mref_V_vph_berkeley(i)
    vsv    = Mref_V_vsv_berkeley(i)
    vsh    = Mref_V_vsh_berkeley(i)
    eta    = Mref_V_eta_berkeley(i)
    Qkappa = Mref_V_Qkappa_berkeley(i)
    Qmu    = Mref_V_Qmu_berkeley(i)
  else
    ! interpolates between one layer below to actual radius layer,
    ! that is from radius_ref(i-1) to r using the values at i-1 and i
    frac = (r-Mref_V_radius_berkeley(i-1))/(Mref_V_radius_berkeley(i)-Mref_V_radius_berkeley(i-1))

    ! interpolated model parameters
    rho = Mref_V_density_berkeley(i-1)  + frac * (Mref_V_density_berkeley(i)- Mref_V_density_berkeley(i-1))
    vpv = Mref_V_vpv_berkeley(i-1)      + frac * (Mref_V_vpv_berkeley(i)    - Mref_V_vpv_berkeley(i-1)    )
    vph = Mref_V_vph_berkeley(i-1)      + frac * (Mref_V_vph_berkeley(i)    - Mref_V_vph_berkeley(i-1)    )
    vsv = Mref_V_vsv_berkeley(i-1)      + frac * (Mref_V_vsv_berkeley(i)    - Mref_V_vsv_berkeley(i-1)    )
    vsh = Mref_V_vsh_berkeley(i-1)      + frac * (Mref_V_vsh_berkeley(i)    - Mref_V_vsh_berkeley(i-1)    )
    eta = Mref_V_eta_berkeley(i-1)      + frac * (Mref_V_eta_berkeley(i)    - Mref_V_eta_berkeley(i-1)    )
    Qkappa = Mref_V_Qkappa_berkeley(i-1)+ frac * (Mref_V_Qkappa_berkeley(i) - Mref_V_Qkappa_berkeley(i-1) )
    Qmu = Mref_V_Qmu_berkeley(i-1)      + frac * (Mref_V_Qmu_berkeley(i)    - Mref_V_Qmu_berkeley(i-1)    )
  endif

  ! make sure Vs is zero in the outer core even if roundoff errors on depth
  ! also set fictitious attenuation to a very high value (attenuation is not used in the fluid)
  if (MIMIC_NATIVE_SPECFEM) then
    if (iregion_code == IREGION_OUTER_CORE) then
      vsv = 0.d0
      vsh = 0.d0
      Qkappa = 3000.d0
      Qmu = 3000.d0
    endif
  endif

  ! non-dimensionalize
  ! time scaling (s^{-1}) is done with scaleval
  scaleval = dsqrt(PI*GRAV*EARTH_RHOAV)

  rho = rho / EARTH_RHOAV
  vpv = vpv / (EARTH_R * scaleval)
  vph = vph / (EARTH_R * scaleval)
  vsv = vsv / (EARTH_R * scaleval)
  vsh = vsh / (EARTH_R * scaleval)

  end subroutine model_1dberkeley

!
!--------------------------------------------------------------------------------------------------
!

  subroutine determine_1dberkeley_moho_layer()

! subroutine to determine the moho layer/node, radius and depth

  use model_1dberkeley_par
  use constants, only: myrank,IIN,IMAIN

  implicit none

  ! local parameter
  double precision, allocatable :: derivdensity(:), derivVp(:), derivVs(:)
  double precision :: dr,maxderivdensity

  double precision, parameter :: tol = 2.0d0**(-5)

  integer :: i,j,ier,totalDiscont

  ! initializes moho node
  moho1D_layer_index = 1

  ! Allocate temporary arrays
  allocate(derivdensity(NR_REF_BERKELEY), derivVp(NR_REF_BERKELEY), derivVs(NR_REF_BERKELEY),stat=ier)
  if (ier /= 0) stop 'Error allocating temporary deriv arrays'

  ! find the discontinuities
  totalDiscont = 0
  do i = 1, NR_REF_BERKELEY-1
    dr = Mref_V_radius_berkeley(i+1) - Mref_V_radius_berkeley(i)

    if (abs(dr) < tol) then
      derivdensity(i) = Mref_V_density_berkeley(i+1) - Mref_V_density_berkeley(i)
      derivVp(i) = Mref_V_vpv_berkeley(i+1) - Mref_V_vpv_berkeley(i)
      derivVs(i) = Mref_V_vsv_berkeley(i+1) - Mref_V_vsv_berkeley(i)

      totalDiscont = totalDiscont + 1
    else
      derivdensity(i) = (Mref_V_density_berkeley(i+1) - Mref_V_density_berkeley(i)) / dr
      derivVp(i) = (Mref_V_vpv_berkeley(i+1) - Mref_V_vpv_berkeley(i)) / dr
      derivVs(i) = (Mref_V_vsv_berkeley(i+1) - Mref_V_vsv_berkeley(i)) / dr
    endif
  enddo

  ! Determine the Mohorovicic discontinuity node
  ! Conditions to select the moho discontinuity:
  ! 1. Radius don't change, hence discontinuity
  ! 2. Vsv(i) and Vsv(i+1) > 0
  ! 3. delta Vsv > 0
  ! 4. max depth of 90km
  ! 5. max density change within 90 km from surface
  maxderivdensity = 0.d0

  j = 1
  do i = 1, NR_REF_BERKELEY-1
    dr = Mref_V_radius_berkeley(i+1) - Mref_V_radius_berkeley(i)

    if (abs(dr) < tol) then
      if ((abs(Mref_V_vsv_berkeley(i)) > tol) &
          .and. (abs(Mref_V_vsv_berkeley(i+1)) > tol) &
          .and. (abs(derivVs(i)) > tol) &
          .and. (abs(Mref_V_radius_berkeley(i)-6371000.) < 90000.)) then

        if (abs(derivdensity(i)) > maxderivdensity) then
          maxderivdensity = abs(derivdensity(i))
          ! new moho layer
          moho1D_layer_index = i
          moho1D_radius = Mref_V_radius_berkeley(i+1) / 1000.d0  ! new moho radius (converted to km)
          moho1D_depth = 6371.d0 - moho1D_radius                 ! new moho depth (in km)
        endif
      endif
      j = j + 1
    endif
  enddo

  ! user output
  if (myrank == 0) then
    write(IMAIN,*) '  1D Moho layer index    = ',moho1D_layer_index
    write(IMAIN,*) '           radius        = ',sngl(moho1D_radius),"(km)"
    write(IMAIN,*) '           depth         = ',sngl(moho1D_depth),"(km)"
    write(IMAIN,*)
    call flush_IMAIN()
  endif

  ! free temporary arrays
  deallocate(derivdensity, derivVp, derivVs)

  end subroutine determine_1dberkeley_moho_layer

!
!--------------------------------------------------------------------------------------------------
!

  !! subroutine to determine moho radius from reference 1D earth model
  subroutine get_1dberkeley_moho_radius(moho_radius)

  use model_1dberkeley_par, only: moho1D_radius

  implicit none

  double precision, intent(out) :: moho_radius

  ! return moho radius from 1D reference model (in km)
  moho_radius = moho1D_radius

  end subroutine get_1dberkeley_moho_radius

!
!--------------------------------------------------------------------------------------------------
!

  subroutine get_1dberkeley_moho_depth(moho_depth)

  use model_1dberkeley_par, only: moho1D_depth

  implicit none
  double precision, intent(out) :: moho_depth

  ! return moho depth from 1D reference model (in km)
  moho_depth = moho1D_depth

  end subroutine get_1dberkeley_moho_depth
