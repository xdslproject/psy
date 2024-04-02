module my_module
contains


  subroutine main()
    integer :: b, i, j, k
    real :: a

    real :: data_1(10,10,10)
    real, dimension(:,:,:), allocatable :: alloc_data

    allocate(alloc_data(10,10,10))

    do i=1, 10
      do j=1, 10
        do k=1, 10
          data_1(k,j,i)=(i*10*10)+(j*10)+k
          alloc_data(k,j,i)=(i*10*10)+(j*10)+k
        end do
      end do
    end do

    do i=1, 10
      do j=1, 10
        do k=1, 10
          a=data_1(k,j,i)
          print *, j, i, k, a
        end do
      end do
    end do
  end subroutine main
end module my_module



!program mpi_test
!use mpi_tester
!implicit none

!  call test()

!end program mpi_test
