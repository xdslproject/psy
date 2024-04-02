subroutine main()
	use timer
    real*8, dimension(1024,1024,1024) :: su, sv, sw, u, v, w
    real*8, dimension(1024) :: tzc1, tzc2, tzd1, tzd2
    integer :: k, j, i, nx, ny, nz
    integer :: itimer0, itimer1
    
    nx=1024
    ny=1024
    nz=1024

    call timer_init()
    call timer_start(itimer0, label='Initialise')
    
    do i=1, nx
      do j=1, ny
        do k=1, nz
          u(k,j,i)=10.0
          v(k,j,i)=20.0
          w(k,j,i)=30.0
        end do
      end do
    end do
    
    do k=1, nz
      tzc1(k)=50.0
      tzc2(k)=15.0
      tzd1(k)=100.0
      tzd2(k)=5.0
    end do

    call timer_stop(itimer0)
    call timer_start(itimer1, label='Compute')
    

    do i=2,nx-1
      do j=2,ny-1
        do k=2,nz-1
          su(k, j, i)=&
            (2.0*(u(k, j, i-1)*(u(k, j, i)+u(k, j, i-1))-u(k, j, i+1)*(u(k, j, i)+u(k, j, i+1)))) + &
            (1.0*(u(k, j-1, i)*(v(k, j-1, i)+v(k, j-1, i+1))-u(k, j+1, i)*(v(k, j, i)+v(k, j, i+1)))) + &
            (tzc1(k)*u(k-1, j, i)*(w(k-1, j, i)+w(k-1, j, i+1))-tzc2(k)*u(k+1, j, i)*(w(k, j, i)+w(k, j, i+1)))

          sv(k, j, i)=&
            (2.0*(v(k, j-1, i)*(v(k, j, i)+v(k, j-1, i))-v(k, j+1, i)*(v(k, j, i)+v(k, j+1, i)))) + &
            (2.0*(v(k, j, i-1)*(u(k, j, i-1)+u(k, j+1, i-1))-v(k, j, i+1)*(u(k, j, i)+u(k, j+1, i)))) + &
            (tzc1(k)*v(k-1, j, i)*(w(k-1, j, i)+w(k-1, j+1, i))-tzc2(k)*v(k+1, j, i)*(w(k, j, i)+w(k, j+1, i)))

          sw(k, j, i)=&
            (tzd1(k)*w(k-1, j, i)*(w(k, j, i)+w(k-1, j, i))-tzd2(k)*w(k+1, j, i)*(w(k, j, i)+w(k+1, j, i))) + &
            (2.0*(w(k, j, i-1)*(u(k, j, i-1)+u(k+1, j, i-1))-w(k, j, i+1)*(u(k, j, i)+u(k+1, j, i)))) + &
            (2.0*(w(k, j-1, i)*(v(k, j-1, i)+v(k+1, j-1, i))-w(k, j+1, i)*(v(k, j, i)+v(k+1, j, i))))
        end do
      end do
    end do

    call timer_stop(itimer1)

    call timer_report()
end subroutine main
