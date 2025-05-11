
real, dimension(N, M) :: A
real, dimension(N) :: y
real, dimension(M) :: x

do j = 1, M
  do i = 1, N
    y(i) = y(i) + A(i, j) * x(j)
  end do
end do
