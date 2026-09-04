!=====================================================================
!
!                       S p e c f e m 3 D  G l o b e
!                       ----------------------------
!
! Serial MPI wrapper routines for the mesher. These routines allow
! xmeshfem3D_globe to run without MPI when the mesh decomposition is truly
! serial, i.e. NCHUNKS = NPROC_XI = NPROC_ETA = 1.
!
!=====================================================================

module my_mpi

  implicit none

  integer, parameter :: serial_comm_world = 0
  integer, parameter :: serial_comm_self = 0
  integer, parameter :: serial_comm_null = -1
  integer, parameter :: serial_info_null = 0
  integer, parameter :: serial_proc_null = -1

  integer :: my_local_mpi_comm_world = serial_comm_world
  integer :: my_local_mpi_comm_for_bcast = serial_comm_null
  integer :: my_local_mpi_comm_inter = serial_comm_null

end module my_mpi

  subroutine init_mpi()

  use my_mpi
  use shared_parameters, only: NUMBER_OF_SIMULTANEOUS_RUNS,BROADCAST_SAME_MESH_AND_MODEL

  implicit none

  integer :: ier

  call open_parameter_file_from_main_only(ier)
  if (ier /= 0) stop 'an error occurred while opening the parameter file'
  call read_value_integer(NUMBER_OF_SIMULTANEOUS_RUNS, 'NUMBER_OF_SIMULTANEOUS_RUNS', ier)
  if (ier /= 0) stop 'Error reading Par_file parameter NUMBER_OF_SIMULTANEOUS_RUNS'
  call read_value_logical(BROADCAST_SAME_MESH_AND_MODEL, 'BROADCAST_SAME_MESH_AND_MODEL', ier)
  if (ier /= 0) stop 'Error reading Par_file parameter BROADCAST_SAME_MESH_AND_MODEL'
  call close_parameter_file()

  call world_split()

  end subroutine init_mpi

  subroutine finalize_mpi()

  implicit none

  call world_unsplit()

  end subroutine finalize_mpi

  subroutine abort_mpi()

  implicit none

  stop 'error, program ended in exit_MPI'

  end subroutine abort_mpi

  subroutine synchronize_all()

  implicit none

  end subroutine synchronize_all

  subroutine synchronize_all_comm(comm)

  implicit none

  integer,intent(in) :: comm
  integer :: unused_i

  unused_i = comm

  end subroutine synchronize_all_comm

  double precision function wtime()

  implicit none

  real :: time

  call cpu_time(time)
  wtime = dble(time)

  end function wtime

  integer function null_process()

  use my_mpi

  implicit none

  null_process = serial_proc_null

  end function null_process

  subroutine test_request(request,flag_result_test)

  implicit none

  integer :: request
  logical :: flag_result_test

  request = 0
  flag_result_test = .true.

  end subroutine test_request

  subroutine wait_req(req)

  implicit none

  integer :: req

  req = 0

  end subroutine wait_req

  logical function is_valid_comm(comm)

  use my_mpi

  implicit none

  integer, intent(in) :: comm

  is_valid_comm = comm /= serial_comm_null

  end function is_valid_comm

  subroutine bcast_iproc_i(buffer,iproc)

  implicit none

  integer :: iproc
  integer :: buffer
  integer :: unused_i

  unused_i = iproc
  unused_i = buffer

  end subroutine bcast_iproc_i

  subroutine bcast_all_i(buffer, countval)

  implicit none

  integer :: countval
  integer, dimension(countval) :: buffer
  integer :: unused_i

  unused_i = countval
  unused_i = buffer(1)

  end subroutine bcast_all_i

  subroutine bcast_all_singlei(buffer)

  implicit none

  integer :: buffer
  integer :: unused_i

  unused_i = buffer

  end subroutine bcast_all_singlei

  subroutine bcast_all_singlel(buffer)

  implicit none

  logical :: buffer
  logical :: unused_l

  unused_l = buffer

  end subroutine bcast_all_singlel

  subroutine bcast_all_cr(buffer, countval)

  use constants, only: CUSTOM_REAL

  implicit none

  integer :: countval
  real(kind=CUSTOM_REAL), dimension(countval) :: buffer
  real(kind=CUSTOM_REAL) :: unused_cr

  unused_cr = buffer(1)

  end subroutine bcast_all_cr

  subroutine bcast_all_singlecr(buffer)

  use constants, only: CUSTOM_REAL

  implicit none

  real(kind=CUSTOM_REAL) :: buffer
  real(kind=CUSTOM_REAL) :: unused_cr

  unused_cr = buffer

  end subroutine bcast_all_singlecr

  subroutine bcast_all_r(buffer, countval)

  implicit none

  integer :: countval
  real, dimension(countval) :: buffer
  real :: unused_r

  unused_r = buffer(1)

  end subroutine bcast_all_r

  subroutine bcast_all_singler(buffer)

  implicit none

  real :: buffer
  real :: unused_r

  unused_r = buffer

  end subroutine bcast_all_singler

  subroutine bcast_all_dp(buffer, countval)

  implicit none

  integer :: countval
  double precision, dimension(countval) :: buffer
  double precision :: unused_dp

  unused_dp = buffer(1)

  end subroutine bcast_all_dp

  subroutine bcast_all_singledp(buffer)

  implicit none

  double precision :: buffer
  double precision :: unused_dp

  unused_dp = buffer

  end subroutine bcast_all_singledp

  subroutine bcast_all_ch(buffer, countval)

  implicit none

  integer :: countval
  character(len=countval) :: buffer
  character(len=countval) :: unused_ch

  unused_ch = buffer

  end subroutine bcast_all_ch

  subroutine bcast_all_ch_array(buffer,ndim,countval)

  implicit none

  integer :: countval,ndim
  character(len=countval),dimension(ndim) :: buffer
  character(len=countval) :: unused_ch

  unused_ch = buffer(1)

  end subroutine bcast_all_ch_array

  subroutine bcast_all_ch_array2(buffer,ndim1,ndim2,countval)

  implicit none

  integer :: countval,ndim1,ndim2
  character(len=countval),dimension(ndim1,ndim2) :: buffer
  character(len=countval) :: unused_ch

  unused_ch = buffer(1,1)

  end subroutine bcast_all_ch_array2

  subroutine bcast_all_l(buffer, countval)

  implicit none

  integer :: countval
  logical,dimension(countval) :: buffer
  logical :: unused_l

  unused_l = buffer(1)

  end subroutine bcast_all_l

  subroutine bcast_all_i_for_database(buffer, countval)

  implicit none

  integer :: countval
  integer :: buffer
  integer :: unused_i

  unused_i = countval
  unused_i = buffer

  end subroutine bcast_all_i_for_database

  subroutine bcast_all_l_for_database(buffer, countval)

  implicit none

  integer :: countval
  logical :: buffer
  integer :: unused_i
  logical :: unused_l

  unused_i = countval
  unused_l = buffer

  end subroutine bcast_all_l_for_database

  subroutine bcast_all_cr_for_database(buffer, countval)

  use constants, only: CUSTOM_REAL

  implicit none

  integer :: countval
  real(kind=CUSTOM_REAL) :: buffer
  integer :: unused_i
  real(kind=CUSTOM_REAL) :: unused_cr

  unused_i = countval
  unused_cr = buffer

  end subroutine bcast_all_cr_for_database

  subroutine bcast_all_dp_for_database(buffer, countval)

  implicit none

  integer :: countval
  double precision :: buffer
  integer :: unused_i
  double precision :: unused_dp

  unused_i = countval
  unused_dp = buffer

  end subroutine bcast_all_dp_for_database

  subroutine bcast_all_r_for_database(buffer, countval)

  implicit none

  integer :: countval
  real :: buffer
  integer :: unused_i
  real :: unused_r

  unused_i = countval
  unused_r = buffer

  end subroutine bcast_all_r_for_database

  subroutine min_all_i(sendbuf, recvbuf)

  implicit none

  integer:: sendbuf, recvbuf

  recvbuf = sendbuf

  end subroutine min_all_i

  subroutine min_all_all_i(sendbuf, recvbuf)

  implicit none

  integer :: sendbuf, recvbuf

  recvbuf = sendbuf

  end subroutine min_all_all_i

  subroutine min_all_cr(sendbuf, recvbuf)

  use constants, only: CUSTOM_REAL

  implicit none

  real(kind=CUSTOM_REAL) :: sendbuf, recvbuf

  recvbuf = sendbuf

  end subroutine min_all_cr

  subroutine min_all_all_cr(sendbuf, recvbuf)

  use constants, only: CUSTOM_REAL

  implicit none

  real(kind=CUSTOM_REAL) :: sendbuf, recvbuf

  recvbuf = sendbuf

  end subroutine min_all_all_cr

  subroutine min_all_all_dp(sendbuf, recvbuf)

  implicit none

  double precision :: sendbuf, recvbuf

  recvbuf = sendbuf

  end subroutine min_all_all_dp

  subroutine max_all_i(sendbuf, recvbuf)

  implicit none

  integer :: sendbuf, recvbuf

  recvbuf = sendbuf

  end subroutine max_all_i

  subroutine max_all_all_veci(buffer,countval)

  implicit none

  integer :: countval
  integer,dimension(countval),intent(inout) :: buffer
  integer :: unused_i

  unused_i = buffer(1)

  end subroutine max_all_all_veci

  subroutine max_all_all_i(val,recvval)

  implicit none

  integer,intent(in) :: val
  integer,intent(out) :: recvval

  recvval = val

  end subroutine max_all_all_i

  subroutine max_all_cr(sendbuf, recvbuf)

  use constants, only: CUSTOM_REAL

  implicit none

  real(kind=CUSTOM_REAL) :: sendbuf, recvbuf

  recvbuf = sendbuf

  end subroutine max_all_cr

  subroutine max_all_all_cr(sendbuf, recvbuf)

  use constants, only: CUSTOM_REAL

  implicit none

  real(kind=CUSTOM_REAL):: sendbuf, recvbuf

  recvbuf = sendbuf

  end subroutine max_all_all_cr

  subroutine max_all_dp(sendbuf, recvbuf)

  implicit none

  double precision :: sendbuf, recvbuf

  recvbuf = sendbuf

  end subroutine max_all_dp

  subroutine any_all_l(sendbuf, recvbuf)

  implicit none

  logical :: sendbuf, recvbuf

  recvbuf = sendbuf

  end subroutine any_all_l

  subroutine sum_all_i(sendbuf, recvbuf)

  implicit none

  integer :: sendbuf, recvbuf

  recvbuf = sendbuf

  end subroutine sum_all_i

  subroutine sum_all_all_i(sendbuf, recvbuf)

  implicit none

  integer :: sendbuf, recvbuf

  recvbuf = sendbuf

  end subroutine sum_all_all_i

  subroutine sum_all_cr(sendbuf, recvbuf)

  use constants, only: CUSTOM_REAL

  implicit none

  real(kind=CUSTOM_REAL) :: sendbuf, recvbuf

  recvbuf = sendbuf

  end subroutine sum_all_cr

  subroutine sum_all_all_cr(sendbuf, recvbuf)

  use constants, only: CUSTOM_REAL

  implicit none

  real(kind=CUSTOM_REAL) :: sendbuf, recvbuf

  recvbuf = sendbuf

  end subroutine sum_all_all_cr

  subroutine sum_all_dp(sendbuf, recvbuf)

  implicit none

  double precision :: sendbuf, recvbuf

  recvbuf = sendbuf

  end subroutine sum_all_dp

  subroutine sum_all_3Darray_dp(sendbuf, recvbuf, nx,ny,nz)

  implicit none

  integer :: nx,ny,nz
  double precision, dimension(nx,ny,nz) :: sendbuf, recvbuf

  recvbuf(:,:,:) = sendbuf(:,:,:)

  end subroutine sum_all_3Darray_dp

  subroutine isend_cr(sendbuf, sendcount, dest, sendtag, req)

  use constants, only: CUSTOM_REAL

  implicit none

  integer :: sendcount, dest, sendtag, req
  real(kind=CUSTOM_REAL), dimension(sendcount) :: sendbuf
  real(kind=CUSTOM_REAL) :: unused_cr
  integer :: unused_i

  unused_cr = sendbuf(1)
  unused_i = dest
  unused_i = sendtag
  req = 0

  end subroutine isend_cr

  subroutine isend_dp(sendbuf, sendcount, dest, sendtag, req)

  implicit none

  integer :: sendcount, dest, sendtag, req
  double precision, dimension(sendcount) :: sendbuf
  double precision :: unused_dp
  integer :: unused_i

  unused_dp = sendbuf(1)
  unused_i = dest
  unused_i = sendtag
  req = 0

  end subroutine isend_dp

  subroutine irecv_cr(recvbuf, recvcount, dest, recvtag, req)

  use constants, only: CUSTOM_REAL

  implicit none

  integer :: recvcount, dest, recvtag, req
  real(kind=CUSTOM_REAL), dimension(recvcount) :: recvbuf
  real(kind=CUSTOM_REAL) :: unused_cr
  integer :: unused_i

  unused_cr = recvbuf(1)
  unused_i = dest
  unused_i = recvtag
  req = 0

  end subroutine irecv_cr

  subroutine irecv_dp(recvbuf, recvcount, dest, recvtag, req)

  implicit none

  integer :: recvcount, dest, recvtag, req
  double precision, dimension(recvcount) :: recvbuf
  double precision :: unused_dp
  integer :: unused_i

  unused_dp = recvbuf(1)
  unused_i = dest
  unused_i = recvtag
  req = 0

  end subroutine irecv_dp

  subroutine recv_i(recvbuf, recvcount, dest, recvtag)

  implicit none

  integer :: dest,recvtag
  integer :: recvcount
  integer,dimension(recvcount) :: recvbuf
  integer :: unused_i

  unused_i = recvbuf(1)
  unused_i = dest
  unused_i = recvtag

  end subroutine recv_i

  subroutine recv_r(recvbuf, recvcount, dest, recvtag)

  implicit none

  integer :: dest,recvtag
  integer :: recvcount
  real,dimension(recvcount) :: recvbuf
  real :: unused_r
  integer :: unused_i

  unused_r = recvbuf(1)
  unused_i = dest
  unused_i = recvtag

  end subroutine recv_r

  subroutine recv_cr(recvbuf, recvcount, dest, recvtag)

  use constants, only: CUSTOM_REAL

  implicit none

  integer :: dest,recvtag
  integer :: recvcount
  real(kind=CUSTOM_REAL),dimension(recvcount) :: recvbuf
  real(kind=CUSTOM_REAL) :: unused_cr
  integer :: unused_i

  unused_cr = recvbuf(1)
  unused_i = dest
  unused_i = recvtag

  end subroutine recv_cr

  subroutine recv_dp(recvbuf, recvcount, dest, recvtag)

  implicit none

  integer :: dest,recvtag
  integer :: recvcount
  double precision,dimension(recvcount) :: recvbuf
  double precision :: unused_dp
  integer :: unused_i

  unused_dp = recvbuf(1)
  unused_i = dest
  unused_i = recvtag

  end subroutine recv_dp

  subroutine recv_ch(recvbuf, recvcount, dest, recvtag)

  implicit none

  integer :: dest,recvtag
  integer :: recvcount
  character(len=recvcount) :: recvbuf
  character(len=recvcount) :: unused_ch
  integer :: unused_i

  unused_ch = recvbuf
  unused_i = dest
  unused_i = recvtag

  end subroutine recv_ch

  subroutine recv_singlei(recvbuf, dest, recvtag)

  implicit none

  integer :: dest,recvtag
  integer :: recvbuf
  integer :: unused_i

  unused_i = recvbuf
  unused_i = dest
  unused_i = recvtag

  end subroutine recv_singlei

  subroutine recv_singlel(recvbuf, dest, recvtag)

  implicit none

  integer :: dest,recvtag
  logical :: recvbuf
  integer :: unused_i
  logical :: unused_l

  unused_l = recvbuf
  unused_i = dest
  unused_i = recvtag

  end subroutine recv_singlel

  subroutine send_ch(sendbuf, sendcount, dest, sendtag)

  implicit none

  integer :: dest,sendtag
  integer :: sendcount
  character(len=sendcount) :: sendbuf
  character(len=sendcount) :: unused_ch
  integer :: unused_i

  unused_ch = sendbuf
  unused_i = dest
  unused_i = sendtag

  end subroutine send_ch

  subroutine send_i(sendbuf, sendcount, dest, sendtag)

  implicit none

  integer :: dest,sendtag
  integer :: sendcount
  integer,dimension(sendcount):: sendbuf
  integer :: unused_i

  unused_i = sendbuf(1)
  unused_i = dest
  unused_i = sendtag

  end subroutine send_i

  subroutine send_singlei(sendbuf, dest, sendtag)

  implicit none

  integer :: dest,sendtag
  integer :: sendbuf
  integer :: unused_i

  unused_i = sendbuf
  unused_i = dest
  unused_i = sendtag

  end subroutine send_singlei

  subroutine send_singlel(sendbuf, dest, sendtag)

  implicit none

  integer :: dest,sendtag
  logical :: sendbuf
  integer :: unused_i
  logical :: unused_l

  unused_l = sendbuf
  unused_i = dest
  unused_i = sendtag

  end subroutine send_singlel

  subroutine send_r(sendbuf, sendcount, dest, sendtag)

  implicit none

  integer :: dest,sendtag
  integer :: sendcount
  real,dimension(sendcount):: sendbuf
  real :: unused_r
  integer :: unused_i

  unused_r = sendbuf(1)
  unused_i = dest
  unused_i = sendtag

  end subroutine send_r

  subroutine send_cr(sendbuf, sendcount, dest, sendtag)

  use constants, only: CUSTOM_REAL

  implicit none

  integer :: dest,sendtag
  integer :: sendcount
  real(kind=CUSTOM_REAL),dimension(sendcount):: sendbuf
  real(kind=CUSTOM_REAL) :: unused_cr
  integer :: unused_i

  unused_cr = sendbuf(1)
  unused_i = dest
  unused_i = sendtag

  end subroutine send_cr

  subroutine send_dp(sendbuf, sendcount, dest, sendtag)

  implicit none

  integer :: dest,sendtag
  integer :: sendcount
  double precision,dimension(sendcount):: sendbuf
  double precision :: unused_dp
  integer :: unused_i

  unused_dp = sendbuf(1)
  unused_i = dest
  unused_i = sendtag

  end subroutine send_dp

  subroutine sendrecv_cr(sendbuf, sendcount, dest, sendtag, &
                         recvbuf, recvcount, source, recvtag)

  use constants, only: CUSTOM_REAL

  implicit none

  integer :: sendcount, recvcount, dest, sendtag, source, recvtag
  real(kind=CUSTOM_REAL), dimension(sendcount) :: sendbuf
  real(kind=CUSTOM_REAL), dimension(recvcount) :: recvbuf
  integer :: unused_i

  recvbuf(1:min(sendcount,recvcount)) = sendbuf(1:min(sendcount,recvcount))
  unused_i = dest
  unused_i = sendtag
  unused_i = source
  unused_i = recvtag

  end subroutine sendrecv_cr

  subroutine sendrecv_dp(sendbuf, sendcount, dest, sendtag, &
                         recvbuf, recvcount, source, recvtag)

  implicit none

  integer :: sendcount, recvcount, dest, sendtag, source, recvtag
  double precision, dimension(sendcount) :: sendbuf
  double precision, dimension(recvcount) :: recvbuf
  integer :: unused_i

  recvbuf(1:min(sendcount,recvcount)) = sendbuf(1:min(sendcount,recvcount))
  unused_i = dest
  unused_i = sendtag
  unused_i = source
  unused_i = recvtag

  end subroutine sendrecv_dp

  subroutine gather_all_i(sendbuf, sendcnt, recvbuf, recvcount, NPROC)

  implicit none

  integer :: sendcnt, recvcount, NPROC
  integer, dimension(sendcnt) :: sendbuf
  integer, dimension(recvcount,0:NPROC-1) :: recvbuf

  recvbuf(1:min(sendcnt,recvcount),0) = sendbuf(1:min(sendcnt,recvcount))

  end subroutine gather_all_i

  subroutine gather_all_all_i(sendbuf, sendcnt, recvbuf, recvcount, NPROC)

  implicit none

  integer :: sendcnt, recvcount, NPROC
  integer, dimension(sendcnt) :: sendbuf
  integer, dimension(recvcount,0:NPROC-1) :: recvbuf

  recvbuf(1:min(sendcnt,recvcount),0) = sendbuf(1:min(sendcnt,recvcount))

  end subroutine gather_all_all_i

  subroutine gather_all_singlei(sendbuf, recvbuf, NPROC)

  implicit none

  integer :: NPROC
  integer :: sendbuf
  integer, dimension(0:NPROC-1) :: recvbuf

  recvbuf(0) = sendbuf

  end subroutine gather_all_singlei

  subroutine gather_all_all_singlei(sendbuf, recvbuf, NPROC)

  implicit none

  integer :: NPROC
  integer :: sendbuf
  integer, dimension(0:NPROC-1) :: recvbuf

  recvbuf(0) = sendbuf

  end subroutine gather_all_all_singlei

  subroutine gather_all_cr(sendbuf, sendcnt, recvbuf, recvcount, NPROC)

  use constants, only: CUSTOM_REAL

  implicit none

  integer :: sendcnt, recvcount, NPROC
  real(kind=CUSTOM_REAL), dimension(sendcnt) :: sendbuf
  real(kind=CUSTOM_REAL), dimension(recvcount,0:NPROC-1) :: recvbuf

  recvbuf(1:min(sendcnt,recvcount),0) = sendbuf(1:min(sendcnt,recvcount))

  end subroutine gather_all_cr

  subroutine gather_all_dp(sendbuf, sendcnt, recvbuf, recvcount, NPROC)

  implicit none

  integer :: sendcnt, recvcount, NPROC
  double precision, dimension(sendcnt) :: sendbuf
  double precision, dimension(recvcount,0:NPROC-1) :: recvbuf

  recvbuf(1:min(sendcnt,recvcount),0) = sendbuf(1:min(sendcnt,recvcount))

  end subroutine gather_all_dp

  subroutine gatherv_all_i(sendbuf, sendcnt, recvbuf, recvcount, recvoffset,recvcounttot, NPROC)

  implicit none

  integer :: sendcnt,recvcounttot,NPROC
  integer, dimension(NPROC) :: recvcount,recvoffset
  integer, dimension(sendcnt) :: sendbuf
  integer, dimension(recvcounttot) :: recvbuf
  integer :: unused_i

  recvbuf(1:min(sendcnt,recvcounttot)) = sendbuf(1:min(sendcnt,recvcounttot))
  unused_i = recvcount(1)
  unused_i = recvoffset(1)

  end subroutine gatherv_all_i

  subroutine gatherv_all_cr(sendbuf, sendcnt, recvbuf, recvcount, recvoffset,recvcounttot, NPROC)

  use constants, only: CUSTOM_REAL

  implicit none

  integer :: sendcnt,recvcounttot,NPROC
  integer, dimension(NPROC) :: recvcount,recvoffset
  real(kind=CUSTOM_REAL), dimension(sendcnt) :: sendbuf
  real(kind=CUSTOM_REAL), dimension(recvcounttot) :: recvbuf
  integer :: unused_i

  recvbuf(1:min(sendcnt,recvcounttot)) = sendbuf(1:min(sendcnt,recvcounttot))
  unused_i = recvcount(1)
  unused_i = recvoffset(1)

  end subroutine gatherv_all_cr

  subroutine gatherv_all_r(sendbuf, sendcnt, recvbuf, recvcount, recvoffset,recvcounttot, NPROC)

  implicit none

  integer :: sendcnt,recvcounttot,NPROC
  integer, dimension(NPROC) :: recvcount,recvoffset
  real, dimension(sendcnt) :: sendbuf
  real, dimension(recvcounttot) :: recvbuf
  integer :: unused_i

  recvbuf(1:min(sendcnt,recvcounttot)) = sendbuf(1:min(sendcnt,recvcounttot))
  unused_i = recvcount(1)
  unused_i = recvoffset(1)

  end subroutine gatherv_all_r

  subroutine all_gather_all_i(sendbuf, recvbuf, NPROC)

  implicit none

  integer :: NPROC
  integer :: sendbuf
  integer, dimension(NPROC) :: recvbuf

  recvbuf(1) = sendbuf

  end subroutine all_gather_all_i

  subroutine all_gather_all_r(sendbuf, sendcnt, recvbuf, recvcnt, recvoffset, dim1, NPROC)

  implicit none

  integer :: sendcnt, dim1, NPROC
  real, dimension(sendcnt) :: sendbuf
  real, dimension(dim1, NPROC) :: recvbuf
  integer, dimension(NPROC) :: recvoffset, recvcnt
  integer :: unused_i

  recvbuf(1:min(sendcnt,dim1),1) = sendbuf(1:min(sendcnt,dim1))
  unused_i = recvcnt(1)
  unused_i = recvoffset(1)

  end subroutine all_gather_all_r

  subroutine all_gather_all_ch(sendbuf, sendcnt, recvbuf, recvcnt, recvoffset, dim1, dim2, NPROC)

  implicit none

  integer :: sendcnt, dim1, dim2, NPROC
  character(len=dim2), dimension(sendcnt) :: sendbuf
  character(len=dim2), dimension(dim1, NPROC) :: recvbuf
  integer, dimension(NPROC) :: recvoffset, recvcnt
  integer :: unused_i

  recvbuf(1:min(sendcnt,dim1),1) = sendbuf(1:min(sendcnt,dim1))
  unused_i = recvcnt(1)
  unused_i = recvoffset(1)

  end subroutine all_gather_all_ch

  subroutine scatter_all_singlei(sendbuf, recvbuf, NPROC)

  implicit none

  integer :: NPROC
  integer, dimension(0:NPROC-1) :: sendbuf
  integer :: recvbuf

  recvbuf = sendbuf(0)

  end subroutine scatter_all_singlei

  subroutine world_size(sizeval)

  implicit none

  integer,intent(out) :: sizeval

  sizeval = 1

  end subroutine world_size

  subroutine world_size_comm(sizeval,comm)

  implicit none

  integer,intent(out) :: sizeval
  integer,intent(in) :: comm
  integer :: unused_i

  sizeval = 1
  unused_i = comm

  end subroutine world_size_comm

  subroutine world_rank(rank)

  implicit none

  integer,intent(out) :: rank

  rank = 0

  end subroutine world_rank

  subroutine world_rank_comm(rank,comm)

  implicit none

  integer,intent(out) :: rank
  integer,intent(in) :: comm
  integer :: unused_i

  rank = 0
  unused_i = comm

  end subroutine world_rank_comm

  subroutine world_duplicate(comm)

  use my_mpi

  implicit none

  integer,intent(out) :: comm

  comm = serial_comm_world

  end subroutine world_duplicate

  subroutine world_get_comm(comm)

  use my_mpi

  implicit none

  integer,intent(out) :: comm

  comm = my_local_mpi_comm_world

  end subroutine world_get_comm

  subroutine world_get_comm_self(comm)

  use my_mpi

  implicit none

  integer,intent(out) :: comm

  comm = serial_comm_self

  end subroutine world_get_comm_self

  subroutine world_get_info_null(info)

  use my_mpi

  implicit none

  integer,intent(out) :: info

  info = serial_info_null

  end subroutine world_get_info_null

  subroutine world_comm_free(comm)

  use my_mpi

  implicit none

  integer,intent(inout) :: comm

  comm = serial_comm_null

  end subroutine world_comm_free

  subroutine world_split()

  use my_mpi
  use constants, only: OUTPUT_FILES_BASE,mygroup
  use shared_parameters, only: NUMBER_OF_SIMULTANEOUS_RUNS, &
    BROADCAST_SAME_MESH_AND_MODEL,OUTPUT_FILES

  implicit none

  if (NUMBER_OF_SIMULTANEOUS_RUNS /= 1) &
    stop 'Error: serial xmeshfem3D_globe requires NUMBER_OF_SIMULTANEOUS_RUNS = 1'

  my_local_mpi_comm_world = serial_comm_world
  my_local_mpi_comm_for_bcast = serial_comm_null
  my_local_mpi_comm_inter = serial_comm_null
  mygroup = 0
  OUTPUT_FILES = OUTPUT_FILES_BASE(1:len_trim(OUTPUT_FILES_BASE))
  BROADCAST_SAME_MESH_AND_MODEL = .false.

  end subroutine world_split

  subroutine world_unsplit()

  implicit none

  end subroutine world_unsplit
