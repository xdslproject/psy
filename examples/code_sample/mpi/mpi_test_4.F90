module mpi_test_4
implicit none
contains

  subroutine main()
    use mpi
    integer :: s(1), d, r, v
  
    call MPI_Init()
  
    call MPI_Finalize()
  end subroutine main
end module mpi_test_4
