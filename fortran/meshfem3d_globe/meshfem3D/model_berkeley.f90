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

!--------------------------------------------------------------------------------------------------
! SEMUCB - WM1
!
! A global radially anisotropic shear-wave speed model developed by French & Romanowicz [2014].
!
! reference:
!   Whole-mantle radially anisotropic shear velocity structure from spectral-element waveform tomography
!   S. W. French,   B. A. Romanowicz
!   Geophysical Journal International, Volume 199, Issue 3, December 2014,
!   Pages 1303--1327, https://doi.org/10.1093/gji/ggu334
!
! note: uses 1D Berkeley model as reference model together with a smooth crustal model
!--------------------------------------------------------------------------------------------------


!--------------------------------------------------------------------------------------------------
! Berkeley 3D model data
!--------------------------------------------------------------------------------------------------

module model_berkeley_par

  use constants, only: A3d_folder  ! Berkeley A3d model folder

  implicit none

  ! spline arrays
  !double precision, dimension(:), allocatable :: aknot,oknot,aknot2,oknot2    ! old: only for getdel()
  double precision, dimension(:,:), allocatable :: knot_coeff,knot_coeff2

  double precision, dimension(:), allocatable :: mdl
  integer, dimension(:), allocatable :: level,level2

  ! spline node radii
  real, dimension(:), allocatable :: kntrad,kntrad_hh     ! real arrays - to avoid conversion in fspl() to float

  ! parameterization
  integer, parameter :: MAXPAR = 4
  integer, dimension(MAXPAR) :: nknotA1 = 0,nknotA2 = 0

  character(1), dimension(:), allocatable :: parblock
  integer :: npar,ndisc,surface,NBPARAM

  logical :: hknots2_exist = .false.,unconformal = .false.

  ! radius of moho discontinuity from reference 1D model (in km)
  double precision :: moho1D_radius = -1.0

  ! work arrays
  double precision, dimension(:), allocatable :: work_dh
  integer, dimension(:), allocatable :: work_kindex

end module model_berkeley_par

!
!--------------------------------------------------------------------------------------------------
!

  subroutine model_berkeley_broadcast()

! standard routine to setup model

  use model_berkeley_par
  use constants, only: myrank,IMAIN,EARTH_R_KM,PI,MAX_STRING_LEN, &
                       THREE_D_MODEL_BERKELEY,THREE_D_MODEL_BERKELEY_AZIM
  use meshfem_models_par, only: THREE_D_MODEL

  implicit none

  integer,parameter :: unit1 = 51,unit2 = 52
  integer :: dum2,i,j,k,n,mdim,ier
  integer :: size_work,size_knots
  double precision, dimension(:), allocatable :: aknot,oknot,aknot2,oknot2  ! temporary for reading
  double precision :: theta,phi
  character :: trash

  character(len=*), parameter :: A3d_dat_tiso       = trim(A3d_folder) // 'A3d.dat'
  character(len=*), parameter :: hknots_dat_tiso    = trim(A3d_folder) // 'hknots.dat'
  character(len=*), parameter :: hknots2_dat_tiso   = trim(A3d_folder) // 'hknots2.dat'

  character(len=*), parameter :: A3d_dat_azim       = trim(A3d_folder) // 'AZIM/A3d.dat'  ! in DATA/SEMUCB_A3d/AZIM/ subfolder
  character(len=*), parameter :: hknots_dat_azim    = trim(A3d_folder) // 'AZIM/hknots.dat'
  character(len=*), parameter :: hknots2_dat_azim   = trim(A3d_folder) // 'AZIM/hknots2.dat'

  character(len=MAX_STRING_LEN) :: A3d_dat,hknots_dat,hknots2_dat

  double precision, parameter :: deg2rad = PI / 180.d0

  ! A3d
  if (THREE_D_MODEL == THREE_D_MODEL_BERKELEY_AZIM) then
    A3d_dat = A3d_dat_azim
    hknots_dat = hknots_dat_azim
    hknots2_dat = hknots2_dat_azim
  else
    A3d_dat = A3d_dat_tiso
    hknots_dat = hknots_dat_tiso
    hknots2_dat = hknots2_dat_tiso
  endif

  ! user info
  if (myrank == 0) then
    write(IMAIN,*) 'broadcast model: SEMUCB Berkeley'
    if (THREE_D_MODEL == THREE_D_MODEL_BERKELEY) &
      write(IMAIN,*) '  transversely isotropic model'
    if (THREE_D_MODEL == THREE_D_MODEL_BERKELEY_AZIM) &
      write(IMAIN,*) '  azimuthal anisotropic model'
    call flush_IMAIN()
  endif

  ! determine moho radius from 1D reference model
  if (myrank == 0) then
    ! gets exact 1D moho radius (in km)
    call get_1dberkeley_moho_radius(moho1D_radius)

    ! adjust radius slightly to be in mantle
    moho1D_radius = moho1D_radius - 0.1d0

    ! use non-dimensionalized moho radius for comparisons
    moho1D_radius = moho1D_radius / EARTH_R_KM

    !debug
    !print *,'debug: [model_berkeley_broadcast] moho radius = ',moho1D_radius
  endif
  ! broadcasts to all other processes
  call bcast_all_singledp(moho1D_radius)

  !
  ! reads 3D model
  !
  if (myrank == 0) then
    ! spline knots
    open(unit1,file=trim(hknots_dat),status='old',iostat=ier)
    if (ier /= 0) then
      print *,'Error opening file: ',trim(hknots_dat)
      stop 'Error opening file hknots.dat'
    endif

    read(unit1,*) nknotA2(1)

    if (allocated(oknot) .or. allocated(aknot) .or. allocated(level)) then
      print *,'[model_berkeley_broadcast] A3d already initiated'
      return
    endif

    allocate(oknot(nknotA2(1)),aknot(nknotA2(1)),level(nknotA2(1)),stat=ier)
    if (ier /= 0) stop 'Error allocating oknot,.. arrays'
    oknot(:) = 0.d0
    aknot(:) = 0.d0
    level(:) = 0

    do i = 1,nknotA2(1)
      read(unit1,*) oknot(i),aknot(i),level(i)
    enddo
    close(unit1)

    inquire(file=trim(hknots2_dat),exist=hknots2_exist)

    if (hknots2_exist) then
      open(unit1,file=trim(hknots2_dat),status='old',iostat=ier)
      if (ier /= 0) then
        print *,'Error opening file: ',trim(hknots2_dat)
        stop 'Error opening file hknots2.dat'
      endif

      read(unit1,*) nknotA2(2)

      if (allocated(oknot2) .or. allocated(aknot2) .or. allocated(level2)) then
        print *,'[model_berkeley_broadcast] A3d hnots2 already initiated'
        return
      endif

      allocate(oknot2(nknotA2(2)),aknot2(nknotA2(2)),level2(nknotA2(2)),stat=ier)
      if (ier /= 0) stop 'Error allocating oknot2,.. arrays'
      oknot2(:) = 0.d0
      aknot2(:) = 0.d0
      level2(:) = 0

      do i = 1,nknotA2(2)
        read(unit1,*) oknot2(i),aknot2(i),level2(i)
      enddo
      close(unit1)
    else
      nknotA2(2) = nknotA2(1)
    endif

    !open model file and read in model
    open(unit2,file=trim(A3d_dat),status='old',iostat=ier)
    if (ier /= 0) then
      print *,'Error opening file: ',trim(A3d_dat)
      stop 'Error opening file A3d.dat'
    endif

    read(unit2,*) npar

    NBPARAM = npar
    if (npar > MAXPAR)   stop 'npar greater than MAXPAR'
    if (npar <= 0)   stop 'npar must be > 0'

    allocate(parblock(npar),stat=ier)
    if (ier /= 0) stop 'Error allocating parblock array'
    parblock(:) = ''

    do i = 1,npar
      read(unit2,*) dum2,nknotA1(i),parblock(i)

      ! same number of splines for all parameters
      if (i > 1 .and. nknotA1(i) /= nknotA1(1)) then
        stop 'Inconsistent A1 splines between parameters'
      endif

      if (i > 2) then
        nknotA2(i) = dum2
        if (nknotA2(i) /= nknotA2(2)) stop 'Param 3 and 4 need the same A2 splines than param 2'
      else if (dum2 /= nknotA2(i)) then
        stop 'Inconsistent hknots.dat and A3d.dat'
      endif
      if (i == 2 .and. nknotA2(i) /= nknotA2(1)) then
        unconformal = .true.
        if (.not. hknots2_exist) stop 'unconformal grid requires hknots2.dat'
      endif
    enddo

    read(unit2,*) ndisc

    surface = 0
    if (ndisc > 0) then
      surface = ndisc
      if (unconformal) print *,'discontinuities assumed same grid as first par'
      do i = 1,surface
        read(unit2,*) dum2, trash
      enddo
    endif

    allocate(kntrad(nknotA1(1)), &
             kntrad_hh(nknotA1(1)-1),stat=ier)
    if (ier /= 0) stop 'Error allocating kntrad,.. arrays'
    kntrad(:) = 0.e0
    kntrad_hh(:) = 0.e0

    read(unit2,*) (kntrad(i),i=1,nknotA1(1))

    ! takes spacings between spline radii
    ! (used for spline evaluations fspl(..) in spl_A3d.c, see routine fill_hh_A3d())
    do i = 1,nknotA1(1)-1
      kntrad_hh(i) = kntrad(i+1) - kntrad(i)
    enddo

    mdim = 0
    do i = 1,npar
      mdim = mdim + nknotA1(i) * nknotA2(i)
    enddo
    mdim = mdim + ndisc * nknotA2(1)

    allocate(mdl(mdim),stat=ier)
    if (ier /= 0) stop 'Error allocating mdl array'
    mdl(:) = 0.d0

    n = 0
    do i = 1,npar
      do j = 1,nknotA1(i)
        read(unit2,*) (mdl(k+n),k=1,nknotA2(i))
        n = n + nknotA2(i)
      enddo
    enddo
    do i = 1,ndisc
      read(unit2,*) (mdl(k+n),k=1,nknotA2(1))
      n = n + nknotA2(1)
    enddo

    if (n /= mdim) stop 'init_A3d dimension error'
    close(unit2)

  endif

  ! user info
  if (myrank == 0) then
    write(IMAIN,*) '  number of parameters: NBPARAM = ',NBPARAM
    call flush_IMAIN()
  endif

  !
  ! broadcast
  !
  !
  call bcast_all_singlei(nknotA2(1))

  if (.not. allocated(oknot)) allocate(oknot(nknotA2(1)))
  if (.not. allocated(aknot)) allocate(aknot(nknotA2(1)))
  if (.not. allocated(level)) allocate(level(nknotA2(1)))

  call bcast_all_i(level,nknotA2(1))
  call bcast_all_dp(oknot,nknotA2(1))
  call bcast_all_dp(aknot,nknotA2(1))

  call bcast_all_singlel(hknots2_exist)

  if (hknots2_exist) then
    call bcast_all_singlei(nknotA2(2))

    if (.not. allocated(oknot2)) allocate(oknot2(nknotA2(2)))
    if (.not. allocated(aknot2)) allocate(aknot2(nknotA2(2)))
    if (.not. allocated(level2)) allocate(level2(nknotA2(2)))

    call bcast_all_i(level2,nknotA2(2))
    call bcast_all_dp(oknot2,nknotA2(2))
    call bcast_all_dp(aknot2,nknotA2(2))
  else
    nknotA2(2) = nknotA2(1)
  endif

  call bcast_all_singlei(npar)
  NBPARAM = npar

  if (.not. allocated(parblock)) allocate(parblock(npar))

  call bcast_all_ch_array(parblock,npar,1)
  call bcast_all_i(nknotA1,MAXPAR)
  call bcast_all_i(nknotA2,MAXPAR)
  call bcast_all_singlel(unconformal)
  call bcast_all_singlei(ndisc)

  ! spline radii
  if (.not. allocated(kntrad)) allocate(kntrad(nknotA1(1)))
  if (.not. allocated(kntrad_hh)) allocate(kntrad_hh(nknotA1(1)-1))

  call bcast_all_r(kntrad,nknotA1(1))
  call bcast_all_r(kntrad_hh,nknotA1(1)-1)

  call bcast_all_singlei(mdim)

  if (.not. allocated(mdl)) allocate(mdl(mdim))

  call bcast_all_dp(mdl,mdim)

  ! allocates temporary work arrays for model_berkeley_shsv() routine
  ! to avoid re-allocations of the same arrays for each call
  size_work = maxval(nknotA2(:))
  allocate(work_dh(size_work),work_kindex(size_work),stat=ier)
  if (ier /= 0) stop 'Error allocating work arrays for Berkeley model'
  work_dh(:) = 0.d0
  work_kindex(:) = 0

  ! great-circle distance evaluations between GLL point position theta/phi and knot position
  ! pre-compute knot position coefficients for getdel_pre()
  size_knots = nknotA2(1)
  allocate(knot_coeff(3,size_knots),stat=ier)
  if (ier /= 0) stop 'Error allocating knot_coeff array for Berkeley model'
  knot_coeff(:,:) = 0.d0
  do i = 1,size_knots
    ! knot coefficients colat/lon in rad
    theta = (90.d0 - aknot(i)) * deg2rad
    phi = oknot(i) * deg2rad
    knot_coeff(1,i) = sin(theta)
    knot_coeff(2,i) = cos(theta)
    knot_coeff(3,i) = phi
  enddo
  ! free unneeded arrays
  deallocate(aknot,oknot)

  if (hknots2_exist) then
    size_knots = nknotA2(2)
    allocate(knot_coeff2(3,size_knots),stat=ier)
    if (ier /= 0) stop 'Error allocating knot_coeff2 array for Berkeley model'
    knot_coeff2(:,:) = 0.d0
    do i = 1,size_knots
      ! knot position colat/lon in rad
      theta = (90.d0 - aknot2(i)) * deg2rad
      phi = oknot2(i) * deg2rad
      knot_coeff2(1,i) = sin(theta)
      knot_coeff2(2,i) = cos(theta)
      knot_coeff2(3,i) = phi
    enddo
    ! free unneeded arrays
    deallocate(aknot2,oknot2)
  endif

  end subroutine model_berkeley_broadcast


!
!--------------------------------------------------------------------------------------------------
!

  subroutine model_berkeley_shsv(r,theta,phi,vpv_ref,vph_ref,vsv_ref,vsh_ref,rho_ref, &
                                 dvsh,dvsv,dvph,dvpv,drho,eta_aniso,iregion_code,CRUSTAL)

! transversely isotropic model
!
! original comment:
!   "returns isotropic vs, vp, and rho assuming scaling dlnVs/dlnVp=2 dlnVs/dlnrho=3
!   also returns anisotropic parameters xi,fi,eta,Gc,Gs,Hc,Hs,Bc,Bs if ifanis=1"
!
! returns perturbations dvpv,dvph,dvsv,dvsh,drho and updated eta_aniso

  use model_berkeley_par
  use constants, only: PI,GRAV,EARTH_RHOAV,EARTH_R,EARTH_R_KM,USE_OLD_VERSION_FORMAT

  implicit none

  double precision, intent(in) :: r,theta,phi
  double precision, intent(in) :: vpv_ref,vph_ref,vsv_ref,vsh_ref,rho_ref
  double precision, intent(out) :: dvsv,dvsh,dvpv,dvph,drho
  double precision, intent(inout) :: eta_aniso

  integer, intent(in) :: iregion_code
  logical, intent(in) :: CRUSTAL

  ! local parameters
  double precision :: x,rho1d,vpv1d,vph1d,vsv1d,vsh1d,eta1d,Qmu1d,Qkappa1d

  double precision :: vs,vp,rho   ! Qs
  double precision :: xi,fi,eta,Gc,Gs
  double precision :: fi_inv

  integer :: jump,effnknot,i,j,k
  integer :: nknots_horiz,nknots_radial
  !integer, dimension(:), allocatable :: kindex

  double precision :: del,dr,dv
  double precision :: AA,CC,FF,LL,NN
  double precision :: eta1,adel1,r_

  ! knot great-circle distances for different spline levels
  !double precision, dimension(8), parameter :: adel = (/63.4, 31.7, 15.8, 7.9, 3.95, 1.98, 0.99/)
  ! Include level 8 (FM, April 2021 - Mod. courtesy of Dan Frost)
  double precision, dimension(8), parameter :: adel = (/63.4, 31.7, 15.8, 7.9, 3.95, 1.98, 0.99, 0.495/)

  !double precision, dimension(:), allocatable :: dh
  !double precision :: lat,lon  ! unused

  double precision :: sin_theta0,cos_theta0
  double precision :: vsv,vsh,vpv,vph,scaleval
  double precision :: aa1, bb1

  double precision,external :: getdel_pre
  double precision,external :: spbsp

  !double precision, parameter :: rad2deg = 180.d0/PI

  ! tolerance for water density check
  double precision, parameter :: TOL_RHO_WATER = 1200.d0 / EARTH_RHOAV   ! non-dimensionalized

  ! initializes model perturbations
  dvsv = 0.d0
  dvsh = 0.d0
  dvpv = 0.d0
  dvph = 0.d0
  drho = 0.d0

  xi = 1.d0
  fi = 1.d0

  !if (.not.model1D_exist) &
  !    stop 'no 1D model file'

  !if (ifanis_berk==1) then
  !    if (.not.(present(xi).and.present(fi).and.present(eta))) &
  !        stop 'A3d_full: ifanis_berk inconsistent'
  !endif

  ! limits radius to stay below 1D moho depth
  ! note: r is non-dimensionalized/normalized input radius between [0,1]
  if (r > moho1D_radius) then
    r_ = moho1D_radius  ! * EARTH_R_KM
    ! re-evaluate reference 1D model values for this updated radius
    call model_1dberkeley(r_,rho1d,vpv1d,vph1d,vsv1d,vsh1d,eta1d,Qkappa1d,Qmu1d,iregion_code,CRUSTAL)
  else
    r_ = r              ! * EARTH_R_KM
    ! reference model values already setup prior to this routine call
    vpv1d = vpv_ref
    vph1d = vph_ref
    vsv1d = vsv_ref
    vsh1d = vsh_ref
    rho1d = rho_ref
    eta1d = eta_aniso
    Qkappa1d = 9999.d0  ! unused
    Qmu1d = 9999.d0     ! unused
  endif

  ! non-dimensionalized radius
  x = r_                ! / EARTH_R_KM

  if (USE_OLD_VERSION_FORMAT) then
    ! old version by mistake compares non-dimensionalized rho1d with density value 1200.d0
    ! thus, this will always be evaluated since rho1d is much smaller than that.
    if (rho1d < 1200.d0) then
      ! No water in RegSEM please
      !
      !debug
      !print *,'debug: rho1d ',rho1d, rho1d * EARTH_RHOAV, rho_ref * EARTH_RHOAV,'radius ',x * EARTH_R_KM

      ! originally, this likely wanted to re-scale the radius to be below ocean at 6368.0 km?
      ! however, this formula won't work - x can still be above ocean radius.
      ! this scaling was moving up the radius value, thus shifting all mantle velocities from the 1D definition slightly up.
      x = r_ * EARTH_R_KM / 6367.999d0
      call model_1dberkeley(x,rho1d,vpv1d,vph1d,vsv1d,vsh1d,eta1d,Qkappa1d,Qmu1d,iregion_code,CRUSTAL)
    endif
  else
    ! fix
    ! check non-dimensionalized density
    if (rho1d < TOL_RHO_WATER) then
      ! No water in RegSEM please
      ! takes radius slightly below ocean at 6368.0 km to get mantle/crust values from the layer below ocean
      x = 6367.999d0 / EARTH_R_KM
      call model_1dberkeley(x,rho1d,vpv1d,vph1d,vsv1d,vsh1d,eta1d,Qkappa1d,Qmu1d,iregion_code,CRUSTAL)
    endif
  endif

  ! below use r_ as radius in km
  r_ = r_ * EARTH_R_KM

  ! re-dimensionalizes 1d model values
  scaleval = EARTH_R * dsqrt(PI*GRAV*EARTH_RHOAV)

  rho1d = rho1d * EARTH_RHOAV
  vpv1d = vpv1d * scaleval
  vph1d = vph1d * scaleval
  vsv1d = vsv1d * scaleval
  vsh1d = vsh1d * scaleval

  rho = rho1d
  AA  = vph1d*vph1d * rho
  CC  = vpv1d*vpv1d * rho
  LL  = vsv1d*vsv1d * rho
  NN  = vsh1d*vsh1d * rho
  FF  = eta1d * (AA - 2.d0 * LL)

  ! unused
  !Qs  = Qmu1d
  !call get_1Dmodel_TI(AA,CC,FF,LL,NN,rho,Qs,r_*1000.d0)
  !if (rho < 1200.d0)   call get_1Dmodel_TI(AA,CC,FF,LL,NN,rho,Qs,6367999.d0)   ! No water in RegSEM please

  eta1 = FF / (AA - 2.d0 * LL)

  ! Voigt average
  vp = sqrt((3.d0 * CC + (8.d0 + 4.d0 * eta1) * AA + 8.d0 * (1.d0 - eta1) * LL) / (15.d0 * rho))
  vs = sqrt((CC + (1.d0 - 2.d0 * eta1) * AA + (6.d0 + 4.d0 * eta1) * LL + 5.d0 * NN) / (15.d0 * rho))

  ! initial values from 1D reference model
  eta = eta1
  xi = NN / LL
  fi = CC / AA

  ! point lat/lon in degrees for getdel()
  !lat = 90.d0 - rad2deg * theta
  !lon = rad2deg * phi

  ! coefficients for getdel_pre()
  sin_theta0 = sin(theta)
  cos_theta0 = cos(theta)

  ! Vs perturbation
  if (r_ > kntrad(nknotA1(1)) .or. r_ < kntrad(1)) then
    dv = 0.d0
  else
    jump = 0

    !allocate(dh(nknotA2(1)),kindex(nknotA2(1)))
    work_dh(:) = 0.d0
    work_kindex(:) = 0
    effnknot = 0

    nknots_horiz = nknotA2(1)
    nknots_radial = nknotA1(1)

    do i = 1,nknots_horiz
      ! gets great-circle distance between position theta/phi and knot
      !old: del = getdel(lat,lon,aknot(i),oknot(i))
      del = getdel_pre(sin_theta0,cos_theta0,phi,knot_coeff(1,i),knot_coeff(2,i),knot_coeff(3,i))

      adel1 = adel(level(i))
      if (del <= adel1 * 2.d0) then
        effnknot = effnknot+1
        work_kindex(effnknot) = i
        work_dh(effnknot) = spbsp(del,adel1)
      endif
    enddo

    dv = 0.d0
    do i = 1,nknots_radial
      ! spline value
      call fspl(i,nknots_radial,kntrad,kntrad_hh,r_,dr)

      do j = 1,effnknot
        dv = dv + dr * work_dh(j) * mdl(jump + work_kindex(j) + nknots_horiz * (i-1))
      enddo
    enddo
    !deallocate(dh,kindex)
  endif

  ! Scaling
  vs = vs + dv * vs
  vp = vp + 0.5d0 * dv * vp
  rho = rho + 0.33333d0 * dv * rho

  ! Perturbation of other parameters
  ! note: default in tiso A3d.dat is npar == 2
  if (npar == 2) then
    ! xi perturbation (dv)
    if (r_ > kntrad(nknotA1(2)) .or. r_ < kntrad(1)) then
      dv = 0.d0
    else
      jump = jump + nknotA1(1) * nknotA2(1)

      work_dh(:) = 0.d0
      work_kindex(:) = 0
      effnknot = 0

      nknots_horiz = nknotA2(2)
      nknots_radial = nknotA1(2)

      if (unconformal) then
        ! unconformal grid
        do i = 1,nknots_horiz
          ! gets great-circle distance between position theta/phi and knot
          !old: del = getdel(lat,lon,aknot2(i),oknot2(i))
          del = getdel_pre(sin_theta0,cos_theta0,phi,knot_coeff2(1,i),knot_coeff2(2,i),knot_coeff2(3,i))

          adel1 = adel(level2(i))
          if (del <= adel1 * 2.d0) then
            effnknot = effnknot + 1
            work_kindex(effnknot) = i
            work_dh(effnknot) = spbsp(del,adel1)
          endif
        enddo
      else
        ! conformal grid
        do i = 1,nknots_horiz
          ! gets great-circle distance between position theta/phi and knot
          !old: del = getdel(lat,lon,aknot(i),oknot(i))
          del = getdel_pre(sin_theta0,cos_theta0,phi,knot_coeff(1,i),knot_coeff(2,i),knot_coeff(3,i))

          adel1 = adel(level(i))
          if (del <= adel1 * 2.d0) then
            effnknot = effnknot + 1
            work_kindex(effnknot) = i
            work_dh(effnknot) = spbsp(del,adel1)
          endif
        enddo
      endif

      dv = 0.d0
      do i = 1,nknots_radial
        ! spline value
        call fspl(i,nknots_radial,kntrad,kntrad_hh,r_,dr)

        do j = 1,effnknot
          dv = dv + dr * work_dh(j) * mdl(jump + work_kindex(j) + nknots_horiz * (i-1))
        enddo
      enddo
    endif

    ! npar == k == 2
    xi = xi + dv * xi
    fi = fi - 1.5d0 * dv * fi
    eta = eta - 2.5d0 * dv * eta

  else if (npar > 2) then
    ! npar > 2
    do k = 2,npar
      if (r_ > kntrad(nknotA1(k)) .or. r_ < kntrad(1)) then
        dv = 0.d0
      else
        jump = jump + nknotA1(k-1) * nknotA2(k-1)

        !allocate(dh(nknotA2(k)),kindex(nknotA2(k)))
        work_dh(:) = 0.d0
        work_kindex(:) = 0
        effnknot = 0

        nknots_horiz = nknotA2(k)
        nknots_radial = nknotA1(k)

        if (unconformal) then
          ! unconformal grid
          do i = 1,nknots_horiz
            ! gets great-circle distance between position theta/phi and knot
            !old: del = getdel(lat,lon,aknot2(i),oknot2(i))
            del = getdel_pre(sin_theta0,cos_theta0,phi,knot_coeff2(1,i),knot_coeff2(2,i),knot_coeff2(3,i))

            adel1 = adel(level2(i))
            if (del <= adel1 * 2.d0) then
              effnknot = effnknot+1
              work_kindex(effnknot) = i
              work_dh(effnknot) = spbsp(del,adel1)
            endif
          enddo
        else
          ! conformal grid
          do i = 1,nknots_horiz
            ! gets great-circle distance between position theta/phi and knot
            !old: del = getdel(lat,lon,aknot(i),oknot(i))
            del = getdel_pre(sin_theta0,cos_theta0,phi,knot_coeff(1,i),knot_coeff(2,i),knot_coeff(3,i))

            adel1 = adel(level(i))
            if (del <= adel1 * 2.d0) then
              effnknot = effnknot+1
              work_kindex(effnknot) = i
              work_dh(effnknot) = spbsp(del,adel1)
            endif
          enddo
        endif

        dv = 0.d0
        do i = 1,nknots_radial
          ! spline value
          call fspl(i,nknots_radial,kntrad,kntrad_hh,r_,dr)

          do j = 1,effnknot
            dv = dv + dr * work_dh(j) * mdl(jump + work_kindex(j) + nknots_horiz * (i-1))
          enddo
        enddo
        !deallocate(dh,kindex)
      endif

      if (k == 2) then
        ! xi perturbation (dv)
        xi = xi + dv * xi
        fi = fi - 1.5d0 * dv * fi
        eta = eta - 2.5d0 * dv * eta
      else if (k == 3) then
        ! Gc perturbation (dv)
        Gc = dv
      else if (k == 4) then
        ! Gs perturbation (dv)
        Gs = dv
      endif
      ! Here we can add a scaling to get Hc, Hs, Bc and Bs
    enddo
  else
    ! no other parameters
    dv = 0.d0
  endif

  ! ============= commented by < FM> on Feb 3, 2020 =====
  !vsv = sqrt(3.d0/(xi+2.d0))*vs
  !vsh = sqrt(xi)*vsv
  !vph = sqrt(5.d0/(fi+4.d0))*vp
  !vpv = sqrt(fi)*vph
  !rho = rho

  ! ====================================================
  ! New conversion relationships < FM> - Feb 3, 2020
  ! Auxiliar values
  fi_inv = 1.d0 / fi
  aa1 = 3.d0 + ( 8.d0 + 4.d0 * eta ) * fi_inv
  bb1 = 1.d0 + ( 1.d0 - 2.d0 * eta ) * fi_inv

  vsv = sqrt( 15.d0 * (vp*vp * bb1 - vs*vs * aa1) / &
              (8.d0 * (1.d0-eta) * bb1 - (6.d0 + 4.d0 * eta + 5.d0 * xi) * aa1 ) )
  vsh = sqrt( xi ) * vsv
  vpv = sqrt( (15.d0 * vp*vp - 8.d0 * (1.d0-eta) * vsv*vsv ) / &
              (3.d0 + ( 8.d0 + 4.d0 * eta ) * fi_inv ) )
  vph = vpv * sqrt( fi_inv )
  rho = rho
  eta_aniso = eta
  ! ===================================================

  ! perturbations
  if (vsv1d == 0.d0) then
    dvsv = 0.d0
  else
    dvsv = vsv / vsv1d - 1.d0
  endif
  if (vsh1d == 0.d0) then
    dvsh = 0.d0
  else
    dvsh = vsh / vsh1d - 1.d0
  endif

  dvpv = vpv / vpv1d - 1.d0
  dvph = vph / vph1d - 1.d0

  drho  = rho / rho1d - 1.d0

  end subroutine model_berkeley_shsv

!
!--------------------------------------------------------------------------------------------------
!

  double precision function getdel_pre(sin_theta0,cos_theta0,phi0,sin_theta,cos_theta,phi)

  use constants, only: PI

  implicit none
  double precision,intent(in) :: sin_theta0,cos_theta0,phi0,sin_theta,cos_theta,phi
  ! local parameters
  double precision :: a
  double precision, parameter :: rad2deg = 180.d0 / PI    ! 180.d0 / (4.d0 * datan(1.d0))

  ! great-circle distance (by law of cosines formula) in degrees
  a = cos_theta * cos_theta0 + sin_theta * sin_theta0 * cos(phi - phi0)

  ! limit to +/- 1 numerical accuracy for acos(..)
  if (abs(a) > 1.d0) a = sign(1.d0,a) * 1.d0

  ! great-circle distance in degrees
  getdel_pre = rad2deg * acos(a)

  end function getdel_pre

!
!--------------------------------------------------------------------------------------------------
!

  double precision function getdel(a0,o0,a,o)

  use constants, only: PI

  implicit none
  double precision,intent(in) :: a0,o0,a,o
  ! local parameters
  double precision :: q0,sq0,cq0
  double precision :: q,sq,cq
  double precision :: ff,cff,arg  ! sff

  double precision, parameter :: deg2rad = PI / 180.d0    ! (4.d0 * datan(1.d0)) / 180.d0
  double precision, parameter :: rad2deg = 180.d0 / PI    ! 180.d0 / (4.d0 * datan(1.d0))

  ! great-circle distance (by law of cosines formula) in degrees
  q0 = (90.d0 - a0) * deg2rad
  sq0 = sin(q0)
  cq0 = cos(q0)

  q = (90.d0 - a) * deg2rad
  sq = sin(q)
  cq = cos(q)

  ff = (o - o0) * deg2rad
  !sff = sin(ff)  ! not used
  cff = cos(ff)

  arg = cq * cq0 + sq * sq0 * cff

  if (arg > 1.d0) arg = 1.d0
  if (arg < -1.d0) arg = -1.d0

  getdel = rad2deg * acos(arg)

  end function getdel

!
!--------------------------------------------------------------------------------------------------
!

  double precision function spbsp(hdel,ahdel)

  implicit none
  double precision,intent(in) :: hdel,ahdel
  ! local parameters
  double precision :: ratio,two_minus_ratio

  ! factor
  ratio = hdel / ahdel

  if (hdel < ahdel) then
    spbsp = (0.75d0 * ratio - 1.5d0) * ratio * ratio + 1.d0
  else if (hdel <= ahdel * 2.d0) then
    two_minus_ratio = 2.d0 - ratio
    if (two_minus_ratio < 0.d0) then
      spbsp = 0.d0
    else
      spbsp = 0.25d0 * two_minus_ratio * two_minus_ratio * two_minus_ratio
    endif
    !if (spbsp < 0.d0) spbsp = 0.d0    ! spbsp is negative if two_minus_ratio is negative
  else
    spbsp = 0.d0
  endif

  end function spbsp

!
!--------------------------------------------------------------------------------------------------
!

  subroutine model_berkeley_3dazim(r,theta,phi,vpv,vph,vsv,vsh,rho,eta_aniso, &
                                   c11,c12,c13,c14,c15,c16,c22,c23,c24,c25,c26, &
                                   c33,c34,c35,c36,c44,c45,c46,c55,c56,c66,iregion_code,CRUSTAL)
! azimuthal anisotropic model
!
! returns updated vpv,vph,vsv,vsh,rho,eta_aniso and full cij

  use model_berkeley_par
  use constants, only: myrank,PI,GRAV,EARTH_RHOAV,EARTH_R,EARTH_R_KM,USE_OLD_VERSION_FORMAT

  implicit none

  double precision, intent(in) :: r,theta,phi
  double precision, intent(inout) :: vpv,vph,vsv,vsh,rho,eta_aniso
  double precision, intent(out):: c11,c12,c13,c14,c15,c16,c22,c23,c24,c25,c26, &
                                  c33,c34,c35,c36,c44,c45,c46,c55,c56,c66
  integer, intent(in) :: iregion_code
  logical, intent(in) :: CRUSTAL

  ! local parameters
  double precision :: x,rho1d,vpv1d,vph1d,vsv1d,vsh1d,eta1d,Qmu1d,Qkappa1d
  double precision :: d11,d12,d13,d14,d15,d16,d22,d23,d24,d25,d26, &
                      d33,d34,d35,d36,d44,d45,d46,d55,d56,d66
  double precision :: A,C,F,L,N,Gc,Gs !,Bc,Bs,Hc,Hs,Ec,Es
  double precision :: vs,vp   ! Qs
  double precision :: xi,fi,eta
  double precision :: fi_inv

  integer :: jump,effnknot,i,j,k
  integer :: nknots_horiz,nknots_radial

  double precision :: del,dr,dv
  double precision :: AA,CC,FF,LL,NN
  double precision :: eta1,adel1,r_

  ! knot great-circle distances for different spline levels
  !double precision, dimension(8), parameter :: adel = (/63.4, 31.7, 15.8, 7.9, 3.95, 1.98, 0.99/)
  ! Include level 8 (FM, April 2021 - Mod. courtesy of Dan Frost)
  double precision, dimension(8), parameter :: adel = (/63.4, 31.7, 15.8, 7.9, 3.95, 1.98, 0.99, 0.495/)

  double precision :: sin_theta0,cos_theta0
  double precision :: vp1d,vs1d ! scaleval
  double precision :: aa1, bb1

  double precision,external :: getdel_pre
  double precision,external :: spbsp

  ! tolerance for water density check
  double precision, parameter :: TOL_RHO_WATER = 1200.d0 / EARTH_RHOAV   ! non-dimensionalized

  ! safety check: for azimuthal parameterization, default is npar == 4 (vs,xi,Gc,Gs)
  if (npar /= 4) call exit_MPI(myrank,'Error: invalid npar in routine model_berkeley_3dazim() for azimuthal model')

  ! initializes model perturbations
  Gc = 0.d0
  Gs = 0.d0
  !Ec = 0.d0  ! not used yet...
  !Es = 0.d0
  !Bc = 0.d0
  !Bs = 0.d0
  !Hc = 0.d0
  !Hs = 0.d0

  xi = 1.d0
  fi = 1.d0

  ! limits radius to stay below 1D moho depth
  ! note: r is non-dimensionalized/normalized input radius between [0,1]
  if (r > moho1D_radius) then
    r_ = moho1D_radius  ! * EARTH_R_KM
    ! re-evaluate reference 1D model values for this updated radius
    call model_1dberkeley(r_,rho1d,vpv1d,vph1d,vsv1d,vsh1d,eta1d,Qkappa1d,Qmu1d,iregion_code,CRUSTAL)
  else
    r_ = r              ! * EARTH_R_KM
    ! reference model values already setup prior to this routine call
    vpv1d = vpv
    vph1d = vph
    vsv1d = vsv
    vsh1d = vsh
    rho1d = rho
    eta1d = eta_aniso
    Qkappa1d = 9999.d0  ! unused
    Qmu1d = 9999.d0     ! unused
  endif

  ! non-dimensionalized radius
  x = r_                ! / EARTH_R_KM

  if (USE_OLD_VERSION_FORMAT) then
    ! old version by mistake compares non-dimensionalized rho1d with density value 1200.d0
    ! thus, this will always be evaluated since rho1d is much smaller than that.
    if (rho1d < 1200.d0) then
      ! No water in RegSEM please
      !
      !debug
      !print *,'debug: rho1d ',rho1d, rho1d * EARTH_RHOAV, rho * EARTH_RHOAV,'radius ',x * EARTH_R_KM

      ! originally, this likely wanted to re-scale the radius to be below ocean at 6368.0 km?
      ! however, this formula won't work - x can still be above ocean radius.
      ! this scaling was moving up the radius value, thus shifting all mantle velocities from the 1D definition slightly up.
      x = r_ * EARTH_R_KM / 6367.999d0
      call model_1dberkeley(x,rho1d,vpv1d,vph1d,vsv1d,vsh1d,eta1d,Qkappa1d,Qmu1d,iregion_code,CRUSTAL)
    endif
  else
    ! fix
    ! check non-dimensionalized density
    if (rho1d < TOL_RHO_WATER) then
      ! No water in RegSEM please
      ! takes radius slightly below ocean at 6368.0 km to get mantle/crust values from the layer below ocean
      x = 6367.999d0 / EARTH_R_KM
      call model_1dberkeley(x,rho1d,vpv1d,vph1d,vsv1d,vsh1d,eta1d,Qkappa1d,Qmu1d,iregion_code,CRUSTAL)
    endif
  endif

  ! below use r_ as radius in km
  r_ = r_ * EARTH_R_KM

  ! not needed, as output c11,.. will again be non-dimensionalized
  !
  ! re-dimensionalizes 1d model values
  !scaleval = EARTH_R * dsqrt(PI*GRAV*EARTH_RHOAV)
  !rho1d = rho1d * EARTH_RHOAV
  !vpv1d = vpv1d * scaleval
  !vph1d = vph1d * scaleval
  !vsv1d = vsv1d * scaleval
  !vsh1d = vsh1d * scaleval

  rho = rho1d
  AA  = vph1d*vph1d * rho
  CC  = vpv1d*vpv1d * rho
  LL  = vsv1d*vsv1d * rho
  NN  = vsh1d*vsh1d * rho
  FF  = eta1d * (AA - 2.d0 * LL)

  ! unused
  !Qs  = Qmu1d
  !call get_1Dmodel_TI(AA,CC,FF,LL,NN,rho,Qs,r_*1000.d0)
  !if (rho < 1200.d0)   call get_1Dmodel_TI(AA,CC,FF,LL,NN,rho,Qs,6367999.d0)   ! No water in RegSEM please

  eta1 = FF / (AA - 2.d0 * LL)

  ! Voigt average
  vp = sqrt((3.d0 * CC + (8.d0 + 4.d0 * eta1) * AA + 8.d0 * (1.d0 - eta1) * LL) / (15.d0 * rho))
  vs = sqrt((CC + (1.d0 - 2.d0 * eta1) * AA + (6.d0 + 4.d0 * eta1) * LL + 5.d0 * NN) / (15.d0 * rho))

  vp1d = vp
  vs1d = vs

  ! initial values from 1D reference model
  eta = eta1
  xi = NN / LL
  fi = CC / AA

  ! point lat/lon in degrees for getdel()
  !lat = 90.d0 - rad2deg * theta
  !lon = rad2deg * phi

  ! coefficients for getdel_pre()
  sin_theta0 = sin(theta)
  cos_theta0 = cos(theta)

  ! ##############
  ! Vs perturbation
  ! ##############
  if (r_ > kntrad(nknotA1(1)) .or. r_ < kntrad(1)) then
    dv = 0.d0
  else
    jump = 0

    !allocate(dh(nknotA2(1)),kindex(nknotA2(1)))
    work_dh(:) = 0.d0
    work_kindex(:) = 0
    effnknot = 0

    nknots_horiz = nknotA2(1)
    nknots_radial = nknotA1(1)

    do i = 1,nknots_horiz
      ! gets great-circle distance between position theta/phi and knot
      !old: del = getdel(lat,lon,aknot(i),oknot(i))
      del = getdel_pre(sin_theta0,cos_theta0,phi,knot_coeff(1,i),knot_coeff(2,i),knot_coeff(3,i))

      adel1 = adel(level(i))
      if (del <= adel1 * 2.d0) then
        effnknot = effnknot+1
        work_kindex(effnknot) = i
        work_dh(effnknot) = spbsp(del,adel1)
      endif
    enddo

    dv = 0.d0
    do i = 1,nknots_radial
      ! spline value
      call fspl(i,nknots_radial,kntrad,kntrad_hh,r_,dr)

      do j = 1,effnknot
        dv = dv + dr * work_dh(j) * mdl(jump + work_kindex(j) + nknots_horiz * (i-1))
      enddo
    enddo
    !deallocate(dh,kindex)
  endif

  ! Scaling
  vs = vs + dv * vs
  vp = vp + 0.5d0 * dv * vp
  rho = rho + 0.33333d0 * dv * rho

  ! Perturbation of other parameters
  ! (assumes npar == 4)
  !
  ! ##############
  ! xi,Gc,Gs
  ! ##############
  do k = 2,npar
    if (r_ > kntrad(nknotA1(k)) .or. r_ < kntrad(1)) then
      dv = 0.d0
    else
      jump = jump + nknotA1(k-1) * nknotA2(k-1)

      work_dh(:) = 0.d0
      work_kindex(:) = 0
      effnknot = 0

      nknots_horiz = nknotA2(k)
      nknots_radial = nknotA1(k)

      if (unconformal) then
        ! unconformal grid
        do i = 1,nknots_horiz
          ! gets great-circle distance between position theta/phi and knot
          !old: del = getdel(lat,lon,aknot2(i),oknot2(i))
          del = getdel_pre(sin_theta0,cos_theta0,phi,knot_coeff2(1,i),knot_coeff2(2,i),knot_coeff2(3,i))

          adel1 = adel(level2(i))
          if (del <= adel1 * 2.d0) then
            effnknot = effnknot+1
            work_kindex(effnknot) = i
            work_dh(effnknot) = spbsp(del,adel1)
          endif
        enddo
      else
        ! conformal grid
        do i = 1,nknots_horiz
          ! gets great-circle distance between position theta/phi and knot
          !old: del = getdel(lat,lon,aknot(i),oknot(i))
          del = getdel_pre(sin_theta0,cos_theta0,phi,knot_coeff(1,i),knot_coeff(2,i),knot_coeff(3,i))

          adel1 = adel(level(i))
          if (del <= adel1 * 2.d0) then
            effnknot = effnknot+1
            work_kindex(effnknot) = i
            work_dh(effnknot) = spbsp(del,adel1)
          endif
        enddo
      endif

      dv = 0.d0
      do i = 1,nknots_radial
        ! spline value
        call fspl(i,nknots_radial,kntrad,kntrad_hh,r_,dr)

        do j = 1,effnknot
          dv = dv + dr * work_dh(j) * mdl(jump + work_kindex(j) + nknots_horiz * (i-1))
        enddo
      enddo
    endif

    ! apply perturbation
    if (k == 2) then
      ! xi perturbation (dv)
      xi = xi + dv * xi
      fi = fi - 1.5d0 * dv * fi
      eta = eta - 2.5d0 * dv * eta
    else if (k == 3) then
      ! Gc perturbation (dv)
      Gc = dv * vs1d
    else if (k == 4) then
      ! Gs perturbation (dv)
      Gs = dv * vs1d
    endif
    ! Here we can add a scaling to get Hc, Hs, Bc and Bs for now we ignore these and 4psi parameters Ec,Es
  enddo

  !debug
  !if (myrank == 0) then
  !  if (Gc /= 0.d0 .or. Gs /= 0.d0) then
  !    print *,'debug: r,theta,phi',r,theta,phi,'iregion',iregion_code,CRUSTAL,'rank',myrank
  !    print *,'debug: Gc Gs',Gc,Gs !,'vs1d fi',vs1d,fi
  !  endif
  !endif

  ! ============= commented by < FM> on Feb 3, 2020 =====
  !vsv = sqrt(3.d0/(xi+2.d0))*vs
  !vsh = sqrt(xi)*vsv
  !vph = sqrt(5.d0/(fi+4.d0))*vp
  !vpv = sqrt(fi)*vph
  !rho = rho

  ! ====================================================
  ! New conversion relationships < FM> - Feb 3, 2020
  ! Auxiliar values
  fi_inv = 1.d0 / fi
  aa1 = 3.d0 + ( 8.d0 + 4.d0 * eta ) * fi_inv
  bb1 = 1.d0 + ( 1.d0 - 2.d0 * eta ) * fi_inv

  vsv = sqrt( 15.d0 * (vp*vp * bb1 - vs*vs * aa1) / &
              (8.d0 * (1.d0-eta) * bb1 - (6.d0 + 4.d0 * eta + 5.d0 * xi) * aa1 ) )
  vsh = sqrt( xi ) * vsv
  vpv = sqrt( (15.d0 * vp*vp - 8.d0 * (1.d0-eta) * vsv*vsv ) / &
              (3.d0 + ( 8.d0 + 4.d0 * eta ) * fi_inv ) )
  vph = vpv * sqrt( fi_inv )
  rho = rho
  eta_aniso = eta
  ! ===================================================

  A = rho * vph**2
  C = rho * vpv**2
  L = rho * vsv**2
  N = rho * vsh**2
  F = eta * (A - 2.d0 * L)

  ! build elastic tensor in local coordinates
  !
  ! note: see also comment in routine rotate_tensor_aniso_to_global() in rotate_tensor.f90:
  !
  !       Going from a local azimuth angle (zeta) measured counter-clockwise from South as used in Chen & Tromp (2007)
  !       to an angle measured clockwise from North (zeta’), would mean zeta = pi - zeta’,
  !       which would involve sign changes on the odd cosine terms and on the even sine terms, i.e.,
  !         Jc’ = - Jc, Kc’ = - Kc, Mc’ = - Mc, Dc’ = - Dc
  !       and
  !         Gs’ = - Gs, Bs’ = - Bs, Hs’ = -Hs, Es’ = - Es,
  !       respectively.
  !
  !       Here, the model assumes that azimuth is measured clockwise from North.
  !       Parameters (A, C, L, N, F, Gs, Gc) are given.
  !       All others (Ec, Bc, Hc and Es, Bs, Hs) as well as (Jc, Kc, Mc, Dc and Js, Ks, Ms, Ds) are zero.
  !
  ! general azimuthal anisotropy
  !d11 = A + Ec + Bc
  !d12 = A - 2.d0 * N - Ec
  !d13 = F + Hc
  !d14 = 0.d0                    ! set to zero: d14 = Ds + 2.d0 * (Js + Ms)
  !d15 = 0.d0                    ! set to zero: d15 = 2.d0 * Jc + Dc
  !d16 = 0.5d0 * Bs + Es         ! Bs' == -Bs, Es' == -Es  -> sign changed in: d16 = -0.5d0 * Bs - Es

  !d22 = A + Ec - Bc
  !d23 = F - Hc
  !d24 = 0.d0                    ! set to zero: d24 = 2.d0 * Js - Ds
  !d25 = 0.d0                    ! set to zero: d25 = 2.d0 * (Jc - Mc) - Dc
  !d26 = 0.5d0 * Bs - Es         ! Bs' == -Bs, Es' == -Es  -> sign changed in: d26 = -0.5d0 * Bs + Es

  !d33 = C
  !d34 = 0.d0                    ! set to zero: d34 = 2.d0 * (Js - Ks)
  !d35 = 0.d0                    ! set to zero: d35 = 2.d0 * (Jc - Kc)
  !d36 = Hs                      ! Hs' == -Hs  -> sign changed in: d36 = -Hs

  !d44 = L - Gc
  !d45 = Gs                      ! Gs' == -Gs  -> sign changed in: d45 = -Gs
  !d46 = 0.d0                    ! set to zero: d46 = Mc - Dc

  !d55 = L + Gc
  !d56 = 0.d0                    ! set to zero: d56 = Ds - Ms
  !d66 = N - Ec

  ! azimuthal anisotropy from (A, C, L, N, F, Gs, Gc)
  d11 = A
  d12 = A - 2.d0 * N
  d13 = F
  d14 = 0.d0
  d15 = 0.d0
  d16 = 0.d0

  d22 = A
  d23 = F
  d24 = 0.d0
  d25 = 0.d0
  d26 = 0.d0

  d33 = C
  d34 = 0.d0
  d35 = 0.d0
  d36 = 0.d0

  d44 = L - Gc
  d45 = Gs                      ! Gs' == -Gs  - > sign changed in: d45 = -Gs
  d46 = 0.d0

  d55 = L + Gc
  d56 = 0.d0
  d66 = N

  ! converts dij to global reference frame cij
  call rotate_tensor_radial_to_global(theta,phi,d11,d12,d13,d14,d15,d16,d22,d23,d24,d25,d26, &
                                      d33,d34,d35,d36,d44,d45,d46,d55,d56,d66, &
                                      c11,c12,c13,c14,c15,c16,c22,c23,c24,c25,c26, &
                                      c33,c34,c35,c36,c44,c45,c46,c55,c56,c66)

  end subroutine model_berkeley_3dazim
