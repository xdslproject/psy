
module my_module
contains

  subroutine init_data(d)
    real, intent(inout) :: d(64,64)

    integer :: i, j

    do i=2, 63
      do j=1, 64
        d(j, i)=0.0
      end do
    end do
    do j=1, 64
      d(j, 1)=1.0
      d(j, 64)=10.0
    end do
  end subroutine init_data


  subroutine main()
    integer :: rank, i,j, it

    real :: dd(64,64), v, bnorm, rnorm, norm

    call init_data(dd)

    bnorm=0.0

    do i=2, 63
      do j=2, 63
        bnorm=bnorm+((dd(j,i)*4-dd(j-1,i)-dd(j+1,i)-dd(j,i-1)-dd(j,i+1)) ** 2)
      end do
    end do

    bnorm=sqrt(bnorm)

    do it=1, 1000
      rnorm=0.0
      do i=2, 63
        do j=2, 63
          rnorm=rnorm+((dd(j,i)*4-dd(j-1,i)-dd(j+1,i)-dd(j,i-1)-dd(j,i+1)) ** 2)
        end do
      end do
      norm=sqrt(rnorm)/bnorm
      print *, "Iteration: ", it, " norm of ", norm

      do i=2, 63
        do j=2, 63
          dd(j,i)=0.25*(dd(j-1,i) + dd(j+1,i) + dd(j,i-1) + dd(j,i+1))
        end do
      end do
    end do     
  end subroutine main

end module my_module
