subroutine main()
  use timer
  integer :: itimer0, itimer1

  call timer_init()
  call timer_start(itimer0, label='Initialise')

  call timer_stop(itimer0)
  call timer_start(itimer1, label='Compute')

  call timer_stop(itimer1)
  call timer_report()
end subroutine main
