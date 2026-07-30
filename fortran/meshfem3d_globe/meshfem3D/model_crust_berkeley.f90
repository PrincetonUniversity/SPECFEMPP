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
!
! Berkeley crust implemented as in csem
!
!--------------------------------------------------------------------------------------------------

module model_crust_berkeley_par

  implicit none

  ! moho model
  real, dimension(:,:), allocatable :: moho_start
  integer :: NBT,NBP
  real :: drin

  ! moho filtering
  real, parameter :: drfiltre = 2.e0

  ! crustal model
  double precision, dimension(:,:), allocatable :: crust_array
  integer, dimension(:), allocatable :: crust_array_ind
  ! number of crustal entries
  integer :: num_crust_array = 0


  ! moho depth from 1D reference model (in km)
  double precision :: moho1D_depth = -1.0d0

end module model_crust_berkeley_par

!
!--------------------------------------------------------------------------------------------------
!

  subroutine model_berkeley_crust_broadcast()

! standard routine to setup model

  use model_crust_berkeley_par
  use constants, only: A3d_folder,myrank,IMAIN,IIN

  implicit none

  character(len=100), parameter :: file_crust = trim(A3d_folder) // 'crust2cru2av_2x2.dat'
  character(len=100), parameter :: file_moho = trim(A3d_folder) // 'crust2moho_2x2.dat'
  integer :: ier

  ! user output
  if (myrank == 0) then
    ! user output
    write(IMAIN,*)
    write(IMAIN,*) 'incorporating crustal model: Berkeley'
    write(IMAIN,*)
    call flush_IMAIN()
  endif

  ! moho data
  open(IIN,file=trim(file_moho),status='old',action='read',iostat=ier)
  if (ier /= 0) then
    print *,'Error opening file: ',trim(file_moho)
    stop 'Error opening file crust2moho_2x2.dat'
  endif
  call read_crustmoho_filtre(IIN)
  close(IIN)

  !drfiltre = dr_

  ! crustal data
  open(IIN,file=trim(file_crust),status='old',action='read',iostat=ier)
  if (ier /= 0) then
    print *,'Error opening file: ',trim(file_crust)
    stop 'Error opening file crust2cru2av_2x2.dat'
  endif
  call read_crust_smooth_variable(IIN)
  close(IIN)

  ! gets moho depth from 1D model
  if (myrank == 0) then
    call get_1dberkeley_moho_depth(moho1D_depth)
  endif
  ! broadcasts to all other processes
  call bcast_all_singledp(moho1D_depth)

  end subroutine model_berkeley_crust_broadcast

!
!--------------------------------------------------------------------------------------------------
!

  subroutine model_berkeley_crust(x,theta,phi,vp,vs,rho,moho,found_crust,elem_in_crust,moho_only)

! returns isotropic crustal velocities

  use model_crust_berkeley_par
  use constants, only: PI,GRAV,EARTH_RHOAV,EARTH_R,EARTH_R_KM, &
    USE_OLD_VERSION_FORMAT

  implicit none

  double precision,intent(in) :: x,theta,phi
  double precision,intent(out) :: vp,vs,rho,moho
  logical,intent(out) :: found_crust
  logical,intent(in) :: elem_in_crust,moho_only

  ! local parameters
  double precision :: vsv,vsh,depth,moho_depth
  double precision :: scaleval,scaleval_inv
  !double precision, parameter :: deg2rad = PI/180.d0, rad2deg = 180.d0/PI

  ! initializes
  vp = 0.d0
  vs = 0.d0
  rho = 0.d0
  moho = 0.d0
  found_crust = .false.

  !theta = (90.d0 - lat) * deg2rad ! assumed lat range: [-90,90]
  !phi = lon * deg2rad ! assumed lon range: [-180,180]

  depth = (1.d0 - x) * EARTH_R_KM

  call get_crust_val_csem(theta,phi,depth,rho,vp,vsv,vsh,moho_depth,moho_only)

  ! using crustal values
  if (USE_OLD_VERSION_FORMAT) then
    ! always uses the crustal values
    found_crust = .true.
  else
    ! only uses the crustal values if point above moho
    ! sets flag if point within crust (depth shallower than moho_depth)
    if (depth <= moho_depth .or. elem_in_crust) found_crust = .true.
  endif

  ! only return moho
  if (moho_only) then
    moho = moho_depth / EARTH_R_KM
    ! all done
    return
  endif

  !
  ! get equivalent isotropic vs
  !
  vs = dsqrt((2.d0 * vsv*vsv + vsh*vsh)/3.d0)

  !
  ! scale values for specfem
  !
  scaleval = dsqrt(PI*GRAV*EARTH_RHOAV)
  scaleval_inv = 1.d0 / (EARTH_R * scaleval)

  vp = vp * scaleval_inv
  vs = vs * scaleval_inv

  rho = rho / EARTH_RHOAV

  moho = moho_depth / EARTH_R_KM

  end subroutine model_berkeley_crust

!
!--------------------------------------------------------------------------------------------------
!

  subroutine model_berkeley_crust_aniso(x,theta,phi,vpv,vph,vsv,vsh,eta_aniso,rho,moho,found_crust,elem_in_crust,moho_only)

! returns anisotropic crustal velocities

  use model_crust_berkeley_par
  use constants, only: PI,GRAV,EARTH_RHOAV,EARTH_R,EARTH_R_KM, &
    USE_OLD_VERSION_FORMAT

  implicit none

  double precision,intent(in) :: x,theta,phi
  double precision,intent(out) :: vpv,vph,vsv,vsh,eta_aniso,rho,moho
  logical,intent(out) :: found_crust
  logical,intent(in) :: elem_in_crust,moho_only

  ! local parameters
  double precision :: vp,depth,moho_depth
  double precision :: scaleval,scaleval_inv
  !double precision, parameter :: deg2rad = PI/180.d0, rad2deg = 180.d0/PI

  ! initializes
  vpv = 0.d0
  vph = 0.d0
  vsv = 0.d0
  vsh = 0.d0
  eta_aniso = 1.d0
  rho = 0.d0
  moho = 0.d0
  found_crust = .false.

  !theta = (90.d0 - lat) * deg2rad ! assumed lat range: [-90,90]
  !phi = lon * deg2rad ! assumed lon range: [-180,180]

  depth = (1.d0 - x) * EARTH_R_KM

  call get_crust_val_csem(theta,phi,depth,rho,vp,vsv,vsh,moho_depth,moho_only)

  ! using crustal values
  if (USE_OLD_VERSION_FORMAT) then
    ! always uses the crustal values
    found_crust = .true.
  else
    ! only uses the crustal values if point above moho
    ! sets flag if point within crust (depth shallower than moho_depth)
    if (depth <= moho_depth .or. elem_in_crust) found_crust = .true.
  endif

  ! only return moho
  if (moho_only) then
    moho = moho_depth / EARTH_R_KM
    ! all done
    return
  endif

  !
  ! scale values for specfem
  !
  scaleval = dsqrt(PI*GRAV*EARTH_RHOAV)
  scaleval_inv = 1.d0 / (EARTH_R * scaleval)

  vph = vp * scaleval_inv
  vpv = vp * scaleval_inv
  vsh = vsh * scaleval_inv
  vsv = vsv * scaleval_inv

  rho = rho / EARTH_RHOAV

  moho = moho_depth / EARTH_R_KM

  end subroutine model_berkeley_crust_aniso

!
!--------------------------------------------------------------------------------------------------
!

  subroutine get_crust_val_csem(theta,phi,z,rho,vp,vsv,vsh,moho_depth,moho_only)

  use model_crust_berkeley_par

  implicit none

  double precision,intent(in) :: theta,phi,z
  double precision,intent(out) :: rho,vp,vsv,vsh,moho_depth
  logical,intent(in) :: moho_only

  ! local parameters
  ! 4-th order GLL positions
  ! pre-computed by calling LGL_NODES(4,1.d-6,xi,wi))
  double precision, parameter :: xi_ref(5) = (/ -1.d0, -0.654653670707977d0, 0.d0, 0.654653670707977d0, 1.d0 /)
  ! pre-computed position range in [0,1]
  double precision, parameter :: xi_norm(5) = ( xi_ref(:) + 1.d0) * 0.5d0

  double precision :: xi(5) !,wi(5)
  double precision :: rho_cr(5),vp_cr(5),vsv_cr(5),vsh_cr(5)
  double precision :: x
  integer :: i

  double precision,external :: moho_filtre
  double precision,external :: lagrange

  ! initialize
  rho = 0.d0
  vp = 0.d0
  vsv = 0.d0
  vsh = 0.d0

  ! get moho depth
  moho_depth = moho1D_depth - moho_filtre(theta,phi)

  !debug
  !print *,"debug: [get_crust_val_csem] Moho depth:",moho_depth, &
  !        "moho1D_depth:",moho1D_depth,"moho_filtre:",moho_filtre(theta,phi)

  ! check if anything further to do or return only moho
  if (moho_only) return

  !
  ! horizontal interpolation for all registered depths
  !
  call crust_bilinear_variable(theta,phi,1,rho_cr(1),vp_cr(1),vsv_cr(1),vsh_cr(1))
  call crust_bilinear_variable(theta,phi,2,rho_cr(2),vp_cr(2),vsv_cr(2),vsh_cr(2))
  call crust_bilinear_variable(theta,phi,3,rho_cr(3),vp_cr(3),vsv_cr(3),vsh_cr(3))
  call crust_bilinear_variable(theta,phi,4,rho_cr(4),vp_cr(4),vsv_cr(4),vsh_cr(4))
  call crust_bilinear_variable(theta,phi,5,rho_cr(5),vp_cr(5),vsv_cr(5),vsh_cr(5))

  !
  ! get GLL nodes position
  !
  ! 4th-order GLL node positions
  !call LGL_NODES(4,1.d-6,xi,wi)
  !xi(:) = moho_depth - (xi(:) + 1.d0) * 0.5d0 * moho_depth

  ! above not needed, will use pre-computed xi_norm array
  xi(:) = moho_depth - xi_norm(:) * moho_depth

  x = z
  if (x > maxval(xi)) x = maxval(xi)
  if (x < minval(xi)) x = minval(xi)

  !
  ! init values
  !
  vp  = 0.d0
  vsv = 0.d0
  vsh = 0.d0
  rho = 0.d0

  !
  ! depth interpolation
  !
  do i = 1,5
    rho = rho + lagrange(i,xi,5,x) * rho_cr(i)
    vp  = vp  + lagrange(i,xi,5,x) * vp_cr(i)
    vsv = vsv + lagrange(i,xi,5,x) * vsv_cr(i)
    vsh = vsh + lagrange(i,xi,5,x) * vsh_cr(i)
  enddo

  rho = rho * 1000.d0
  vp  = vp  * 1000.d0
  vsv = vsv * 1000.d0
  vsh = vsh * 1000.d0

  end subroutine get_crust_val_csem

!
!--------------------------------------------------------------------------------------------------
!

  double precision function lagrange(ind,xi,nxi,x)

  implicit none

  integer,intent(in) :: ind,nxi
  double precision,intent(in) :: xi(nxi),x

  ! local parameters
  double precision :: x0
  integer :: i

  ! initializes
  lagrange = 1.d0

  x0 = xi(ind)

  if (nxi == 5) then
    ! NGLL == 5 case
    select case(ind)
    case (1)
      lagrange = (x-xi(2))/(x0-xi(2)) * (x-xi(3))/(x0-xi(3)) * (x-xi(4))/(x0-xi(4)) * (x-xi(5))/(x0-xi(5))
    case (2)
      lagrange = (x-xi(1))/(x0-xi(1)) * (x-xi(3))/(x0-xi(3)) * (x-xi(4))/(x0-xi(4)) * (x-xi(5))/(x0-xi(5))
    case (3)
      lagrange = (x-xi(1))/(x0-xi(1)) * (x-xi(2))/(x0-xi(2)) * (x-xi(4))/(x0-xi(4)) * (x-xi(5))/(x0-xi(5))
    case (4)
      lagrange = (x-xi(1))/(x0-xi(1)) * (x-xi(2))/(x0-xi(2)) * (x-xi(3))/(x0-xi(3)) * (x-xi(5))/(x0-xi(5))
    case (5)
      lagrange = (x-xi(1))/(x0-xi(1)) * (x-xi(2))/(x0-xi(2)) * (x-xi(3))/(x0-xi(3)) * (x-xi(4))/(x0-xi(4))
    end select
  else
    ! general NGLL case
    do i = 1,nxi
      if (i /= ind) lagrange = lagrange * ( (x-xi(i))/(x0-xi(i)) )
    enddo
  endif

  end function lagrange

!
!--------------------------------------------------------------------------------------------------
!

  subroutine LGL_NODES(N,EPS,XI,WI)

  implicit none

  INTEGER :: N ! POLYNOMIAL ORDER
  DOUBLE PRECISION :: EPS ! DESIRED ERROR

  DOUBLE PRECISION :: XI(N+1) ! GAUSS-LOBATTO-LEGENDRE INTERPOLATION POINTS
  DOUBLE PRECISION :: WI(N+1) ! GAUSS-LOBATTO-LEGENDRE QUADRATURE WEIGHT

  DOUBLE PRECISION :: P(N+1,N+1),XOLD(N+1)
  DOUBLE PRECISION :: ERMAX
  INTEGER :: I,J,N1
  DOUBLE PRECISION, PARAMETER :: PI = 3.141592653589793d0  ! PI = 4.D0 * DATAN(1.D0)

  N1 = N+1
  do I = 1,N1
    XI(I) = -COS(PI * real(I-1)/real(N1-1))
  enddo

  ERMAX = 2.d0 * EPS

  DO WHILE(ERMAX > EPS)
    do I = 1,N1
      XOLD(I) = XI(I)
      P(I,1) = 1.d0
      P(I,2) = XI(I)
    enddo
    do J = 2,N
      do I = 1,N1
        P(I,J+1) = ((2*J-1) * XI(I) * P(I,J) - (J-1) * P(I,J-1)) / REAL(J)
      enddo
    enddo
    do I = 1,N1
      XI(I) = XOLD(I) - (XI(I) * P(I,N1) - P(I,N)) / (N1 * P(I,N1))
    enddo
    ERMAX = 0.d0
    do I = 1,N1
      ERMAX = MAX(ERMAX, abs(XI(I)-XOLD(I)))
    enddo
  enddo

  do I = 1,N1
    WI(I) = 2.d0 / (N * N1 * P(I,N1)**2)
  enddo

  end subroutine LGL_NODES

!
!--------------------------------------------------------------------------------------------------
!

  double precision function moho_filtre(theta,phi)

!theta phi en radians
!dr en degre
!reponse en metre

  use model_crust_berkeley_par

  use constants, only: RADIANS_TO_DEGREES
  implicit none

  double precision,intent(in) :: theta,phi
  ! local parameter
  real :: t,p
  real, parameter :: pi = 3.141592653589793e0  ! pi = 4.0 * atan(1.0)
  real, parameter :: deg2rad = pi/180.e0

  t = sngl(theta * RADIANS_TO_DEGREES)
  p = sngl(phi * RADIANS_TO_DEGREES)

  moho_filtre = dble(gauss_filtre1(moho_start,t,p,drfiltre))

  contains

    !-----------------------------------------------------------------
    real function gauss_filtre1(tin,theta,phi,dr)
    !-----------------------------------------------------------------
    implicit none
    real, intent(in) :: theta,phi,dr
    real, dimension(:,:), intent(in) :: tin
    ! local parameters
    real :: thetar,phir
    real :: inte,rfac
    real :: tmp,tmpnorm
    integer :: i,ii,j,jj

    integer, parameter :: LARG = 10

    tmp = 0.e0
    tmpnorm = 0.e0

    do i = 1,LARG+1
       do j = 1,LARG+1
          call get_indexloc(phi,theta,i,j,dr,LARG,ii,jj,phir,thetar)

          ! integration factor
          rfac = ( dr/real(LARG/2) * deg2rad )**2
          inte = cos_cylindre(theta,phi,dr,thetar,phir) * rfac * sin(thetar * deg2rad)

          ! contribution
          tmp = tmp + tin(ii,jj) * inte
          tmpnorm = tmpnorm + inte
       enddo
    enddo

    ! normalizes
    gauss_filtre1 = tmp / tmpnorm

    end function gauss_filtre1

    !----------------------------------------------------------------------
    real function cos_cylindre(t0_,p0_,d0_,theta_,phi_)
    !----------------------------------------------------------------------
    implicit none
    real, intent(in) :: t0_,p0_
    real, intent(in) :: d0_,theta_,phi_
    ! local parameters
    real :: t0,p0,d0,theta,phi,d_ang
    real :: cosa

    ! target location
    t0 = t0_ * deg2rad
    p0 = p0_ * deg2rad

    ! raster location
    theta = theta_ * deg2rad
    phi = phi_ * deg2rad

    d0 = d0_ * deg2rad

    ! distance angulaire au centre du cylindre:
    cosa = cos(theta) * cos(t0) + sin(theta) * sin(t0) * cos(phi - p0)

    if (cosa >= 1.0) then
      d_ang = 0.e0
    else if (cosa <= -1.0) then
      d_ang = pi
    else
      !d_ang = acos(cos(theta) * cos(t0) + sin(theta) * sin(t0) * cos(phi - p0))
      d_ang = acos( cosa )
    endif

    if (d_ang > d0) then
      cos_cylindre = 0.e0
    else
      cos_cylindre = 0.5e0 * (1.e0 + cos(pi * d_ang/d0))
    endif

    end function cos_cylindre

  end function moho_filtre

!
!--------------------------------------------------------------------------------------------------
!

  subroutine get_indexloc(phi,theta,i,j,dr,LARG,ii,jj,phir,thetar)

  use model_crust_berkeley_par

  implicit none
  real, intent(in) :: theta,phi,dr
  integer, intent(in) :: i,j,LARG
  real, intent(out) :: thetar,phir
  integer, intent(out) :: ii,jj

  ! local parameters
  real :: t,p
  real, parameter :: eps = 1.e-8

  p  = phi + (i-1-LARG/2) * dr/real(LARG/2)
  t  = theta + (j-1-LARG/2) * dr/real(LARG/2)

  if (p < 0.e0 - eps) p = p + 360.e0
  if (p >= 360.e0 - eps) p = p - 360.e0

  if (t > 180.e0 - eps) then
    t = t - 180.e0
    p = 360.e0 - p
  else if (t < 0.e0 - eps) then
    t = 180.e0 + t
    p = 360.e0 - p
  endif

  if (p < 0.e0 - eps) p = p + 360.e0
  if (p >= 360.e0 - eps) p = p - 360.e0

  ii = nint(p/drin) + 1
  if (ii > NBP) ii = NBP

  jj = nint(t/drin) + 1
  if (jj > NBT) jj = NBT

  thetar = t
  phir  = p

  end subroutine get_indexloc

!
!--------------------------------------------------------------------------------------------------
!

  subroutine crust_bilinear_variable(theta,phi,gll_ind,rho_crust,vp_crust,vsv_crust,vsh_crust)

!theta phi en radians
!dr en degre
!reponse en metre

  use model_crust_berkeley_par
  use constants, only: PI

  implicit none
  integer, intent(in) :: gll_ind
  double precision, intent(in) :: theta,phi
  double precision, intent(out) :: rho_crust,vp_crust,vsv_crust,vsh_crust

  ! local parameters
  double precision :: dx,dy,t,p,factor
  integer :: j,m
  double precision, parameter :: rad2deg = 180.d0/PI  ! deg2rad = PI/180.d0

  ! sfrench 20110103 model grid spacing (degrees)
  double precision, parameter :: D_DEG = 2.d0
  double precision, parameter :: D_DEG_INV = 1.d0 / D_DEG

  ! latitude in degrees
  t = 90.d0 - theta * rad2deg

  ! sfrench 20110103 new bounds on latitude in accord with new model grid spacing
  !if (t>89.d0)  t = 89.d0
  !if (t <-89.d0) t = -89.d0
  if (t > 90.d0 - D_DEG) t = 90.d0 - D_DEG
  if (t < D_DEG - 90.d0) t = D_DEG - 90.d0

  ! longitude in degrees
  p = phi * rad2deg
  if (p > 180.d0) p = p - 360.d0


  m = 0

  rho_crust = 0.d0
  vp_crust = 0.d0
  vsv_crust = 0.d0
  vsh_crust = 0.d0

  do j = 1,num_crust_array
    ! m must be < 4
    if (m >= 4) exit

    if (gll_ind == crust_array_ind(j)) then
      dy = dabs(crust_array(1,j) - t)
      dy = dy * D_DEG_INV ! sfrench 20110103 : normalized to model grid spacing

      if (dy < 1.d0) then
        dx = dabs(crust_array(2,j) - p)
        if (dx > 180.d0) dx = 360.d0 - dx

        dx = dx * D_DEG_INV ! sfrench 20110103 : normalized to model grid spacing

        if (dabs(dx) < 1.d0) then
          ! Increment number of found locations. Must be <= 4
          m = m + 1

          factor = (1.d0-dx) * (1.d0-dy)

          rho_crust = rho_crust + factor * crust_array(3,j)
          vp_crust  = vp_crust  + factor * crust_array(4,j)
          vsv_crust = vsv_crust + factor * crust_array(5,j)
          vsh_crust = vsh_crust + factor * crust_array(6,j)

          !  print *,'m= ',m,'dx= ',dx,' dy= ',dy, 'Vsv= ',vsv_crust
        endif
      endif
    endif

  enddo

  end subroutine crust_bilinear_variable

!
!--------------------------------------------------------------------------------------------------
!

  subroutine read_crust_smooth_variable(unit)

  use model_crust_berkeley_par

  implicit none
  integer, intent(in) :: unit
  ! local parameters
  integer :: j,ier
  double precision :: t,p,rho,vp,vsv,vsh
  integer :: gll_ind

  read(unit,*) num_crust_array

  ! allocates crustal array
  ! note: file format is: #t #p #gll_ind #rho #vp #vsv #vsh
  allocate(crust_array(6,num_crust_array), &
           crust_array_ind(num_crust_array),stat=ier)
  if (ier /= 0) stop 'Error allocating crust_array,..'
  crust_array(:,:) = -1000.0
  crust_array_ind(:) = 0

  do j = 1,num_crust_array
    ! format: #t #p #gll_ind #rho #vp #vsv #vsh
    read(unit,*) t,p,gll_ind,rho,vp,vsv,vsh

    ! store values
    ! GLL point index
    crust_array_ind(j) = gll_ind + 1  ! GLL point index in range [1,5]

    ! point values
    crust_array(1,j) = t              ! latitude in degree [-90,90]
    crust_array(2,j) = p              ! longitude in degree [-180,180]
    crust_array(3,j) = rho
    crust_array(4,j) = vp
    crust_array(5,j) = vsv
    crust_array(6,j) = vsh
  enddo

  !debug
  !print *,'read_crust_smooth_variable: I have read ',num_crust_array,' crustal inputs!'

  end subroutine read_crust_smooth_variable

!
!--------------------------------------------------------------------------------------------------
!

  subroutine read_crustmoho_filtre(unit)

  use model_crust_berkeley_par

  implicit none
  integer, intent(in) :: unit
  integer :: i,j

  read(unit,*) NBP,NBT,drin

  ! checks
  if (drin /= 2.) STOP 'read_crust_filtre: dr muste be == 2'

  ! allocates moho array
  allocate(moho_start(NBP,NBT))
  moho_start(:,:) = -1000.0

  ! reads in values
  do j = 1,NBP
    do i = 1,NBT
      read(unit,*) moho_start(j,i)
    enddo
  enddo

  ! convert to km
  moho_start(:,:) = moho_start(:,:) / 1000.0

  NBT = NBT - 1

  end subroutine read_crustmoho_filtre
