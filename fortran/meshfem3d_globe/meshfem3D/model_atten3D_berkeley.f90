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
! SEMUCB - WM1 (elastic part)
!! A global radially anisotropic shear-wave speed model developed by French & Romanowicz [2014].
!
! reference:
!   Whole-mantle radially anisotropic shear velocity structure from spectral-element waveform tomography
!   S. W. French,   B. A. Romanowicz
!   Geophysical Journal International, Volume 199, Issue 3, December 2014,
!   Pages 1303--1327, https://doi.org/10.1093/gji/ggu334
!
! note: uses 1D Berkeley model as reference model together with a smooth crustal model
!--------------------------------------------------------------------------------------------------
! anelastic 3D model: SEMUCB-UMQ
! H. Karaoglu and B. Romanowicz
!  Geophysical Journal International,  2018
!
!--------------------------------------------------------------------------------------------------
! Berkeley 3D model data
!--------------------------------------------------------------------------------------------------

module model_atten3D_berkeley_par

  use constants, only: A3d_folder

  implicit none

  ! attenuation model folder
  character(len=*), parameter :: A3dq_folder = trim(A3d_folder) // "3DQ/"     ! folder "DATA/SEMUCB_A3d/3DQ/"

  ! spline arrays
  !double precision, dimension(:), allocatable :: aknot,oknot,aknot2,oknot2    ! old: only for getdelq()
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

end module model_atten3D_berkeley_par

!
!--------------------------------------------------------------------------------------------------
!

  subroutine model_atten3D_berkeley_broadcast()

! standard routine to setup model

  use model_atten3D_berkeley_par
  use constants, only: myrank,IMAIN,EARTH_R_KM,PI

  implicit none

  integer,parameter :: unit1 = 51,unit2 = 52
  integer :: dum2,i,j,k,n,mdim,ier
  integer :: size_work,size_knots
  double precision, dimension(:), allocatable :: aknot,oknot !,aknot2,oknot2  ! temporary for reading
  double precision :: theta,phi
  character :: trash

  character(len=*), parameter :: A3dQ_dat    = trim(A3dq_folder) // 'A3d_Q.dat'
  character(len=*), parameter :: hknotsq_dat = trim(A3dq_folder) // 'hknotsq.dat'
  !character(len=*), parameter :: hknots2_dat = trim(A3dq_folder) // 'hknots2.dat'

  double precision, parameter :: deg2rad = PI / 180.d0

  ! user info
  if (myrank == 0) then
    write(IMAIN,*) 'broadcast model: SEMUCB-3DQ Berkeley'
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
  ! reads 3D attenuation model
  !
  if (myrank == 0) then
    ! spline knots
    open(unit1,file=trim(hknotsq_dat),status='old',iostat=ier)
    if (ier /= 0) then
      print *,'Error opening file: ',trim(hknotsq_dat)
      stop 'Error opening file hknotsq.dat'
    endif

    read(unit1,*) nknotA2(1)

    ! checks if already setup
    if (allocated(oknot) .or. allocated(aknot) .or. allocated(level)) then
      ! user info
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

!   inquire(file=trim(hknots2_dat),exist=hknots2_exist)

!   if (hknots2_exist) then
!     open(unit1,file=trim(hknots2_dat),status='old',iostat=ier)
!     if (ier /= 0) then
!       print *,'Error opening file: ',trim(hknots2_dat)
!       stop 'Error opening file hknots2.dat'
!     endif

!     read(unit1,*) nknotA2(2)

!     if (allocated(oknot2) .or. allocated(aknot2) .or. allocated(level2)) then
!       print *,'[model_berkeley_broadcast] A3d hnots2 already initiated'
!       return
!     endif

!     allocate(oknot2(nknotA2(2)),aknot2(nknotA2(2)),level2(nknotA2(2)),stat=ier)
!     if (ier /= 0) stop 'Error allocating oknot2,.. arrays'
!     oknot2(:) = 0.d0
!     aknot2(:) = 0.d0
!     level2(:) = 0

!     do i = 1,nknotA2(2)
!       read(unit1,*) oknot2(i),aknot2(i),level2(i)
!     enddo
!     close(unit1)
!   else
!     nknotA2(2) = nknotA2(1)
!   endif

    !open model file and read in model
    open(unit2,file=trim(A3dQ_dat),status='old',iostat=ier)
    if (ier /= 0) then
      print *,'Error opening file: ',trim(A3dQ_dat)
      stop 'Error opening file A3d_Q.dat'
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
        stop 'Inconsistent hknots.dat and A3d_Q.dat'
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
    ! (used for spline evaluations fspl(..) in spl__A3d.c, see routine fill_hh_A3d())
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

!  if (hknots2_exist) then
!    call bcast_all_singlei(nknotA2(2))

!    if (.not. allocated(oknot2)) allocate(oknot2(nknotA2(2)))
!    if (.not. allocated(aknot2)) allocate(aknot2(nknotA2(2)))
!    if (.not. allocated(level2)) allocate(level2(nknotA2(2)))

!    call bcast_all_i(level2,nknotA2(2))
!    call bcast_all_dp(oknot2,nknotA2(2))
!    call bcast_all_dp(aknot2,nknotA2(2))
!  else
!    nknotA2(2) = nknotA2(1)
!  endif

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
  ! pre-compute knot position coefficients for getdelq_preq()
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

! if (hknots2_exist) then
!   size_knots = nknotA2(2)
!   allocate(knot_coeff2(3,size_knots),stat=ier)
!   if (ier /= 0) stop 'Error allocating knot_coeff2 array for Berkeley model'
!   knot_coeff2(:,:) = 0.d0
!   do i = 1,size_knots
!     ! knot position colat/lon in rad
!     theta = (90.d0 - aknot2(i)) * deg2rad
!     phi = oknot2(i) * deg2rad
!     knot_coeff2(1,i) = sin(theta)
!     knot_coeff2(2,i) = cos(theta)
!     knot_coeff2(3,i) = phi
!   enddo
    ! free unneeded arrays
!   deallocate(aknot2,oknot2)
! endif

  end subroutine model_atten3D_berkeley_broadcast


!
!--------------------------------------------------------------------------------------------------
!

  subroutine model_atten3D_berkeley(r,theta,phi,rho_ref,Qkappa,Qmu,iregion_code,CRUSTAL)

!  reads perturbations in attenuation dq and returns absolute Qkappa,Qmu (Qkappa stays 1D)

! returns isotropic vs, vp, and rho assuming scaling dlnVs/dlnVp=2 dlnVs/dlnrho=3
! also returns anisotropic parameters xi,fi,eta,Gc,Gs,Hc,Hs,Bc,Bs if ifanis=1

  use model_atten3D_berkeley_par
  use constants

  implicit none

  double precision, intent(in) :: r,theta,phi,rho_ref
! double precision, intent(out) :: dq
  double precision, intent(inout) :: Qmu,Qkappa
!  double precision, intent(in) :: vpv_ref,vph_ref,vsv_ref,vsh_ref,rho_ref
! double precision, intent(out) :: dvsv,dvsh,dvpv,dvph,drho
! double precision, intent(inout) :: eta_aniso

  integer, intent(in) :: iregion_code
  logical, intent(in) :: CRUSTAL

  ! local parameters
  double precision :: x,rho1d,vpv1d,vph1d,vsv1d,vsh1d,eta1d,Qmu1d,Qkappa1d

! double precision :: vs,vp,rho   ! Qs
! double precision :: xi,fi,eta,Gc,Gs
  double precision :: xi,fi
! double precision :: fi_inv
  double precision :: dq

  integer :: jump,effnknot,i,j !,k
  integer :: nknots_horiz,nknots_radial
  !integer, dimension(:), allocatable :: kindex

  double precision :: del,dr,r_
! double precision :: AA,CC,FF,LL,NN
! double precision :: eta1,adel1,r_

  ! knot great-circle distances for different spline levels
  !double precision, dimension(8), parameter :: adel = (/63.4, 31.7, 15.8, 7.9, 3.95, 1.98, 0.99/)
  ! Include level 8 (FM, April 2021 - Mod. courtesy of Dan Frost)
  double precision, dimension(8), parameter :: adel = (/63.4, 31.7, 15.8, 7.9, 3.95, 1.98, 0.99, 0.495/)

  !double precision, dimension(:), allocatable :: dh
  !double precision :: lat,lon  ! unused

  double precision :: sin_theta0,cos_theta0
! double precision :: vsv,vsh,vpv,vph,scaleval
! double precision :: aa1, bb1

  double precision,external :: getdelq_preq
  double precision,external :: spbspq

  !double precision, parameter :: rad2deg = 180.d0/PI

  ! tolerance for water density check
  double precision, parameter :: TOL_RHO_WATER = 1200.d0 / EARTH_RHOAV   ! non-dimensionalized

  ! initializes model perturbations
! dvsv = 0.d0
! dvsh = 0.d0
! dvpv = 0.d0
! dvph = 0.d0
! drho = 0.d0
  dq   = 0.d0
  Qmu1d = Qmu
  Qkappa1d = Qkappa

  xi = 1.d0
  fi = 1.d0

  !if (.not.model1D_exist) &
  !    stop 'no 1D model file'

  !if (ifanis_berk==1) then
  !    if (.not.(present(xi).and.present(fi).and.present(eta))) &
  !        stop 'A3d_full: ifanis_berk inconsistent'
  !endif

  ! no 3D Q in crust for now

  ! limits radius to stay below 1D moho depth
  ! note: r is non-dimensionalized/normalized input radius between [0,1]
  if (r > moho1D_radius) then
    r_ = moho1D_radius  ! * EARTH_R_KM
    ! re-evaluate reference 1D model values for this updated radius
    call model_1dberkeley(r_,rho1d,vpv1d,vph1d,vsv1d,vsh1d,eta1d,Qkappa,Qmu,iregion_code,CRUSTAL)
  else
    r_ = r              ! * EARTH_R_KM
    ! reference model values already setup prior to this routine call
!   vpv1d = vpv_ref
!   vph1d = vph_ref
!   vsv1d = vsv_ref
!   vsh1d = vsh_ref
    rho1d = rho_ref
!   eta1d = eta_aniso
    Qkappa1d = Qkappa
    Qmu1d = Qmu
  endif
! print *,'before fix r_,Qmu,Qkappa Qmu1d Qkappa1d',r_*EARTH_R_KM,Qmu,Qkappa,Qmu1d,Qkappa1d
  ! non-dimensionalized radius
  x = r_                ! / EARTH_R_KM

  if (USE_OLD_VERSION_FORMAT) then
    !debug
    !print *,"debug: model_atten3D_berkeley: using old version format - I am in this part"

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
  !debug
  !print *,'after fix r_,Qmu,Qkappa,Qmu1d,Qkappa1d',r_*EARTH_R_KM,Qmu,Qkappa,Qmu1d,Qkappa1d

  ! below use r_ as radius in km
  r_ = r_ * EARTH_R_KM

  ! coefficients for getdelq_preq()
  sin_theta0 = sin(theta)
  cos_theta0 = cos(theta)

  ! Qs perturbation
  if (r_ > kntrad(nknotA1(1)) .or. r_ < kntrad(1)) then
    dq = 0.d0
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
      !old: del = getdelq(lat,lon,aknot(i),oknot(i))
      del = getdelq_preq(sin_theta0,cos_theta0,phi,knot_coeff(1,i),knot_coeff(2,i),knot_coeff(3,i))

      if (del <= adel(level(i)) * 2.d0) then
        effnknot = effnknot+1
        work_kindex(effnknot) = i
        work_dh(effnknot) = spbspq(del,adel(level(i)))
      endif
    enddo

    dq = 0.d0
    do i = 1,nknots_radial
      ! spline value
      call fspl(i,nknots_radial,kntrad,kntrad_hh,r_,dr)

      do j = 1,effnknot
        dq = dq + dr * work_dh(j) * mdl(jump + work_kindex(j) + nknots_horiz * (i-1))
      enddo
    enddo
    !deallocate(dh,kindex)
  endif
 ! endif ! if below moho

! convert to absolute values of Qmu
! model SEMUCB_UMQ is given in dqmu =10*d(1/(2*qmu) in a3d format
! so that Q3D = Qmu1d+ dq/5.

  Qmu = Qmu1d - dq/5. * Qmu1d*Qmu1d

!  print *,'r qmu1d qmu',r_,Qmu1d,Qmu

  end subroutine model_atten3D_berkeley

!
!--------------------------------------------------------------------------------------------------
!

  double precision function getdelq_preq(sin_theta0,cos_theta0,phi0,sin_theta,cos_theta,phi)

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
  getdelq_preq = rad2deg * acos(a)

  end function getdelq_preq

!
!--------------------------------------------------------------------------------------------------
!

  double precision function getdelq(a0,o0,a,o)

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

  getdelq = rad2deg * acos(arg)

  end function getdelq

!
!--------------------------------------------------------------------------------------------------
!

  double precision function spbspq(hdel,ahdel)

  implicit none
  double precision,intent(in) :: hdel,ahdel
  ! local parameters
  double precision :: ratio,two_minus_ratio

  ! factor
  ratio = hdel / ahdel

  if (hdel < ahdel) then
    spbspq = (0.75d0 * ratio - 1.5d0) * ratio * ratio + 1.d0
  else if (hdel <= ahdel * 2.d0) then
    two_minus_ratio = 2.d0 - ratio
    if (two_minus_ratio < 0.d0) then
      spbspq = 0.d0
    else
      spbspq = 0.25d0 * two_minus_ratio * two_minus_ratio * two_minus_ratio
    endif
    !if (spbspq < 0.d0) spbsp = 0.d0    ! spbsp is negative if two_minus_ratio is negative
  else
    spbspq = 0.d0
  endif

  end function spbspq
