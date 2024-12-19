
subroutine create_directory_if_doesnt_exist(directory)

    use shared_parameters, only: MAX_STRING_LEN

    implicit none
    character(len=MAX_STRING_LEN), intent(in) :: directory
    integer :: status
    logical :: directory_exists

    call system('[[ ! -e ' // directory // ' ]] && mkdir ' // directory)

end subroutine create_directory_if_doesnt_exist
