module mpi_test_2
implicit none

  type, public :: MPI_Request
  end type MPI_Request
contains

  subroutine main()
    use mpi
    integer :: s(1), d, r, v
  
  
    type(MPI_Request) :: request_handle, handles(10)
  
    r= MPI_CommRank()
  
    call MPI_Send(d, 1, 1, 0)
  
    if (r==0) then
      d=12
      call MPI_Isend(d, 1, 1, 0, handles(0))
      call MPI_Waitall(handles, 1)
    else
      call MPI_Irecv(s, 1, 0, 0, request_handle)
      call MPI_Wait(request_handle)
      v=s(1)
      print *, v
    end if
    !call MPI_Waitall(handles, 2)
  end subroutine main
end module mpi_test_2
