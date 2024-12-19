
subroutine print_help_message()

    implicit none

    write(*,*) ' '
    write(*,*) 'Usage: meshfem2D [options]'
    write(*,*) ' '
    write(*,*) 'Options:'
    write(*,*) ' '
    write(*,*) '  -h, --help'
    write(*,*) '     Print this help message'
    write(*,*) ' '
    write(*,*) '  -p, --Par_File'
    write(*,*) '     Specify the parameter file to use'
    write(*,*) ' '

end subroutine print_help_message

subroutine parse_command_line_arguments()

  use constants, only: MAX_STRING_LEN, one
  use shared_parameters, only: Par_file

  implicit none

  integer :: i = 1, n_args

  character(len=MAX_STRING_LEN) :: arg

  ! get the number of command line arguments
  n_args = command_argument_count()

  if (n_args == 0) then
    ! print help message
    call print_help_message()
    ! stop the code
    call stop_the_code('')
  endif

  ! loop over the command line arguments
  do while(.TRUE.)
    ! get the i-th command line argument
    call get_command_argument(i, arg)

    select case (arg)
      case ('-h', '--help')
        ! print help message
        call print_help_message()
        ! stop the code gracefully
        call finalize_mpi()
        call EXIT(0)
      case ('-p', '--Par_File')
        if (i == n_args) then
          ! print error message
          call stop_the_code('Error: missing command line argument for option '//trim(arg))
        endif
        ! get the next command line argument
        call get_command_argument(i+1, Par_file)
        ! skip the next command line argument
        i = i + 1
      case default
        ! print error message
        call stop_the_code('Error: unknown command line argument: '//trim(arg))
    end select

    if (i == n_args) exit
    ! increment the counter
    i = i + 1
  end do

end subroutine parse_command_line_arguments
