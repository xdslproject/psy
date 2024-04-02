subroutine main()
  use mpi
  integer :: d(10), rank, recv(10), v, i

  rank=MPI_CommRank()
  if (rank == 0) then
    do i=1, 10
      d(i)=i
    end do
    call MPI_Send(d, 10, 1, 0)
    print *, rank, "sent"
  end if

  if (rank == 1) then
    call MPI_Recv(recv, 10, 0, 0)
    print *, rank, "recvd"
    do i=1, 10
      v=recv(i)
      print *, "Data ", i, v
    end do
    print *, rank, "done"
  end if
end subroutine main
