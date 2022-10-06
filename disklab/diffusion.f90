!==============================================================
!            MODULE FOR 3-D DIFFUSION EQUATION SOLVER
!
! Two versions of the equations can be solved:
!
!  - The diffusion equation for radiative transfer with
!    Eddington factors
!
!  - The advection-diffusion equation in general, but without
!    Eddington factors
!
!                 C.P. Dullemond, December 2000
!                  Revised version April 2002
!                 Revised version November 2017
!             Converted from f77 to f90 November 2017
!             Time-dependence included November 2017 (***IN PROGRESS***)
!              Advection-Diffusion part August 2019
!
!==============================================================


!==============================================================
!                            PART:
!                DIFFUSION WITH EDDINGTON FACTORS
! 
! This is a general purpose module for setting up and solving
! a 3-D/2-D/1-D diffusion type problem of the form:
!
!      dJ     D  /    1    D(f^ij*J) \
!    g -- - ---- | ------- --------- |  =  source - beta*J
!      dt   Dx^j \ alpha_j    Dx^i   /
!
! J is the mean-intensity, f^ij the Eddington tensor, alpha_j
! the extinction coefficient, source the source term, beta
! the loss coefficient, and the factor g in front of the
! time-derivative is used to include the heat storage capacity
! of the underlying medium (it is g = 1/f_c with f_c defined
! in Kuiper, Klahr, Dullemond, Kley & Henning (2010). Note that
! The E_R in Kuiper et al is E_R = 4*pi*J/c. 
!
! The equations are solved in conservative form, i.e. the flux
!    
!               1    D(f^ij*J) 
!    H^j = - ------- --------- 
!            alpha_j    Dx^i   
!
! is differenced using a numerically conservative scheme with
! surface and volume elements.
!
! The boundary conditions are:
!
!    p (H~)^i + q J  = r
!
! Here (H~)^i are the local (i.e. non-covariant) fluxes:  
! (H~)^i = H^i * sqrt(g_ii). 
!
! The D/Dx^i are the covariant derivatives in a curved
! coordinate system (though space remains flat). The coordinate
! system must be orthogonal (the metric g_ij=diagonal). Three
! coordinate systems are supported:
!
!  - Cartesian              g_ij = diag(1,1,1)
!  - Spherical              g_ij = diag(1,(x^1)^2,(x^1)^2*sin^2(x^2))
!  - Cylindrical            g_ij = diag(1,1,(x^1)^2)
!
! The coordinates x^1,x^2,x^3 are then:
!
!                          x^1        x^2        x^3
!    Cartesian              x          y          z
!    Spherical              R        Theta       Phi
!    Cylindrical            R          z         Phi
!
! One can also solve the 2-D problem (x^1,x^2) or the 1-D 
! problem (x^1), by putting the grid sizes in these extra
! dimensions to zero. But one cannot do (x^1,x^3) or (x^2,x^3).
!
! NOTE: This code solves for the mean intensity J, not for the 
!       gas temperature. So especially for the time-dependent
!       models it is important to set diff_j(:,:,:) as initial
!       condition, and read out diff_j(:,:,:) as the result.
!       EXCEPT when you set usetgas=.true. in the arguments
!       of solve_diffusion(): Then the initial guess of the
!       diff_j is computed from the values you put into the
!       diff_tgas, and at the end the diff_tgas is computed
!       from the diff_j. So if you set usetgas=.true. then
!       it is essentially tgas that is solved. 
!
!--------------------------------------------------------------
!
! HOW TO USE (example):
!
!  nx = 4
!  ny = 4
!  nz = 4
!  call diff_init(METRIC_XYZ,nx,ny,nz)   ! Make a 4x4x4 cartesian grid
!  !
!  ! Set x, y, and z grid
!  !
!  do ix=1,nx
!     diff_x(ix) = -2.0 + 4.0*(ix-1)/(nx-1.0)
!  enddo
!  do iy=1,ny
!     diff_y(iy) = -2.0 + 4.0*(iy-1)/(ny-1.0)
!  enddo
!  do iz=1,nz
!     diff_z(iz) = -2.0 + 4.0*(iz-1)/(nz-1.0)
!  enddo
!  !
!  ! Set the alpha value everywhere
!  !
!  do ix=1,nx
!     do iy=1,ny
!        do iz=1,nz
!           diff_ax(ix,iy,iz) = 1.0
!           diff_ay(ix,iy,iz) = 1.0
!           diff_az(ix,iy,iz) = 1.0
!        enddo
!     enddo
!  enddo
!  !
!  ! Set the source value everywhere
!  !
!  do ix=1,nx
!     do iy=1,ny
!        do iz=1,nz
!           diff_s(ix,iy,iz) = 1.0
!        enddo
!     enddo
!  enddo
!  !
!  ! The boundary conditions of the type p grad J + q J = r
!  !
!  do iy=1,ny
!     do iz=1,nz
!        diff_pxl(iy,iz) = -sqrt(3.)
!        diff_qxl(iy,iz) = 1.0
!        diff_rxl(iy,iz) = 0.0
!        diff_pxr(iy,iz) = sqrt(3.)
!        diff_qxr(iy,iz) = 1.0
!        diff_rxr(iy,iz) = 0.0
!     enddo
!  enddo
!  do ix=1,nx
!     do iz=1,nz
!        diff_pyl(ix,iz) = -sqrt(3.)
!        diff_qyl(ix,iz) = 1.0
!        diff_ryl(ix,iz) = 0.0
!        diff_pyr(ix,iz) = sqrt(3.)
!        diff_qyr(ix,iz) = 1.0
!        diff_ryr(ix,iz) = 0.0
!     enddo
!  enddo
!  do ix=1,nx
!     do iy=1,ny
!        diff_pzl(ix,iy) = -sqrt(3.)
!        diff_qzl(ix,iy) = 1.0
!        diff_rzl(ix,iy) = 0.0
!        diff_pzr(ix,iy) = sqrt(3.)
!        diff_qzr(ix,iy) = 1.0
!        diff_rzr(ix,iy) = 0.0
!     enddo
!  enddo
!  !
!  ! Call the diffusion solver code
!  !
!  call solve_diffusion(convcrit,itermax,.false.)
!  !
!  ! Save the results
!  !
!  ##################
!  !
!  ! Deallocate everything
!  !
!  call diff_finish()
!
!==============================================================

module diffusion
  !
  ! Constants
  !
  integer, parameter :: NBMAX = 9
  integer, parameter :: ZBMAX = 3
  integer, parameter :: ICONVCRIT = 1
  integer, parameter :: METRIC_XYZ    = 1
  integer, parameter :: METRIC_RTHPHI = 2
  integer, parameter :: METRIC_RZPHI  = 3
  integer, parameter :: METRIC_RZPHI_ZCONICAL  = 4
  double precision, parameter :: pisigsb = 55404.34639419066d0  ! = pi/sigma_SB
  doubleprecision,parameter :: diff_tmin = 1.d-4     ! Lower bound to temperature
  !
  ! Global numbers
  !
  integer :: diff_imetric,diff_nband
  integer :: diff_idim,diff_lda,diff_nx,diff_ny,diff_nz
  doubleprecision :: diff_sgnx,diff_sgny,diff_sgnz
  doubleprecision :: diff_yl,diff_yr,diff_zl,diff_zr
  logical :: diff_dimy,diff_dimz,diff_timedep,diff_silent=.false.
  doubleprecision :: diff_fluxcons_error=0.0
  doubleprecision :: diff_mateq_error=0.0
  integer :: diff_mateq_iter=0
  !
  ! Arrays to be filled by the user:
  !
  doubleprecision, allocatable :: diff_x(:)          ! The x-coordinate array
  doubleprecision, allocatable :: diff_y(:)          ! The y-coordinate array
  doubleprecision, allocatable :: diff_z(:)          ! The z-coordinate array
  doubleprecision, allocatable :: diff_wxx(:,:,:)    ! The xx of the traceless edd tensor
  doubleprecision, allocatable :: diff_wxy(:,:,:)    ! The xy of the traceless edd tensor
  doubleprecision, allocatable :: diff_wyy(:,:,:)    ! The yy of the traceless edd tensor
  doubleprecision, allocatable :: diff_wxz(:,:,:)    ! The xz of the traceless edd tensor
  doubleprecision, allocatable :: diff_wyz(:,:,:)    ! The yz of the traceless edd tensor
  !                                                  ! The zz follows from wxx+wyy+wzz=0
  doubleprecision, allocatable :: diff_pxl(:,:)      ! p of left x BC
  doubleprecision, allocatable :: diff_qxl(:,:)      ! q of left x BC
  doubleprecision, allocatable :: diff_rxl(:,:)      ! r of left x BC
  doubleprecision, allocatable :: diff_pxr(:,:)      ! p of right x BC
  doubleprecision, allocatable :: diff_qxr(:,:)      ! q of right x BC
  doubleprecision, allocatable :: diff_rxr(:,:)      ! r of right x BC
  doubleprecision, allocatable :: diff_pyl(:,:)      ! p of left y BC 
  doubleprecision, allocatable :: diff_qyl(:,:)      ! q of left y BC 
  doubleprecision, allocatable :: diff_ryl(:,:)      ! r of left y BC 
  doubleprecision, allocatable :: diff_pyr(:,:)      ! p of right y BC
  doubleprecision, allocatable :: diff_qyr(:,:)      ! q of right y BC
  doubleprecision, allocatable :: diff_ryr(:,:)      ! r of right y BC
  doubleprecision, allocatable :: diff_pzl(:,:)      ! p of left z BC 
  doubleprecision, allocatable :: diff_qzl(:,:)      ! q of left z BC 
  doubleprecision, allocatable :: diff_rzl(:,:)      ! r of left z BC 
  doubleprecision, allocatable :: diff_pzr(:,:)      ! p of right z BC
  doubleprecision, allocatable :: diff_qzr(:,:)      ! q of right z BC
  doubleprecision, allocatable :: diff_rzr(:,:)      ! r of right z BC
  doubleprecision, allocatable :: diff_s(:,:,:)      ! Source term
  doubleprecision, allocatable :: diff_ax(:,:,:)     ! Alpha (in x-direction)
  doubleprecision, allocatable :: diff_ay(:,:,:)     ! Alpha (in y-direction)
  doubleprecision, allocatable :: diff_az(:,:,:)     ! Alpha (in z-direction)
  doubleprecision, allocatable :: diff_b(:,:,:)      ! Beta factor in equation
  doubleprecision, allocatable :: diff_cvrho(:,:,:)  ! (for time-dep) c_v*rho
  doubleprecision, allocatable :: diff_t_irrad(:,:,:)! (for time-dep) equil temp by irradiation
  !
  ! For the advection-diffusion equation only:
  !
  doubleprecision, allocatable :: diff_g(:,:,:)      ! Weighting factor in adv-diff equation
  doubleprecision, allocatable :: diff_vx(:,:,:)     ! Velocity (in x-direction)
  doubleprecision, allocatable :: diff_vy(:,:,:)     ! Velocity (in y-direction)
  doubleprecision, allocatable :: diff_vz(:,:,:)     ! Velocity (in z-direction)
  doubleprecision, allocatable :: diff_dcx(:,:,:)    ! Diff coef (in x-direction)
  doubleprecision, allocatable :: diff_dcy(:,:,:)    ! Diff coef (in y-direction)
  doubleprecision, allocatable :: diff_dcz(:,:,:)    ! Diff coef (in z-direction)
  !
  ! The to-be-solved function J(x,y,z), also serves as initial condition
  !
  doubleprecision, allocatable :: diff_j(:,:,:)      ! Mean intensity
  !
  ! Internally computed arrays
  !
  doubleprecision, allocatable :: diff_hx(:,:,:)     ! Flux H_x
  doubleprecision, allocatable :: diff_hy(:,:,:)     ! Flux H_y
  doubleprecision, allocatable :: diff_hz(:,:,:)     ! Flux H_z
  doubleprecision, allocatable :: diff_fxx(:,:,:)    ! The xx of the full covariant edd tensor
  doubleprecision, allocatable :: diff_fyy(:,:,:)    ! The yy of the full covariant edd tensor
  doubleprecision, allocatable :: diff_fzz(:,:,:)    ! The zz of the full covariant edd tensor
  doubleprecision, allocatable :: diff_fxy(:,:,:)    ! The xy of the full covariant edd tensor
  doubleprecision, allocatable :: diff_fxz(:,:,:)    ! The xz of the full covariant edd tensor
  doubleprecision, allocatable :: diff_fyz(:,:,:)    ! The yz of the full covariant edd tensor
  doubleprecision, allocatable :: diff_error(:,:,:)  ! The error at each point
  doubleprecision, allocatable :: diff_acp(:,:,:)    ! The band matrix elements
  doubleprecision, allocatable :: diff_rhs(:)        ! The right-hand-side of the equation
  doubleprecision, allocatable :: diff_sol(:)        ! The solution vector (normalized)
  integer, allocatable :: diff_mixb(:)               ! Helper index array for band matrix
  doubleprecision, allocatable :: diff_tgas(:,:,:)   ! (for time-dep) Gas/dust temperature
  doubleprecision, allocatable :: diff_gfact(:,:,:)  ! (for time-dep) The g-factor (see above)

contains
  
  !---------------------------------------------------------------
  !                         INIT
  !
  ! One can limit the calculation to 2-D or 1-D by setting nz=1
  ! (2-D) or nz=1 AND ny=1 (1-D). The routine will then 
  ! automatically switch to 2-D or 1-D mode. And if so, then it
  ! will ignore arrays that are not relevant (for instance, fxz
  ! is not relevant for 2-D, nor is the z-coordinate array). 
  !
  ! The Eddington factors f^xx,f^xy,f^yy,f^xz,f^yz,f^zz are given 
  ! to this subroutine in their traceless form: w^xx,w^xy,w^yy,w^xz,
  ! w^yz,w^zz:
  !
  !   f^ij  =  w^ij  +  (1/3) delta^ij
  !
  ! where delta^ij=0 for j.ne.i and delta^ij=1 for i.eq.j. The
  ! w^ij obey:
  !
  !   w^xx + w^yy + w^zz = 0  
  !
  ! so that it follows that
  !
  !   f^xx + f^yy + f^zz = 1
  !
  ! which is required by the formalism of moments.
  !
  ! ARGUMENTS:
  !    imetric       Metric type (see defines)
  !    nx,ny,nz      Grid size in all directions
  !                  If all .gt.1     ----------->  3-D
  !                  If nz=1          ----------->  2-D
  !                  If nz=1,ny=1     ----------->  1-D                        
  !
  !---------------------------------------------------------------
  subroutine diff_init(imetric,nx,ny,nz,timedep)
    implicit none
    integer :: imetric
    integer :: nx,ny,nz
    logical, optional :: timedep
    !
    ! Copy
    !
    diff_nx      = nx
    diff_ny      = ny
    diff_nz      = nz
    diff_lda     = nx*ny*nz
    if(present(timedep)) then
       diff_timedep = timedep
    else
       diff_timedep = .false.    ! Default: Stationary
    endif
    !
    ! Check the dimension
    !
    if(nx.lt.1) then
       write(*,*) 'Error: nx.lt.1'
       stop 13
    endif
    if(ny.lt.1) then
       write(*,*) 'Error: ny.lt.1'
       stop 13
    endif
    if(nz.lt.1) then
       write(*,*) 'Error: nz.lt.1'
       stop 13
    endif
    if(nx.eq.1) then
       write(*,*) 'Sorry, x-dimension cannot be switched off'
       stop 13
    endif
    if((nz.gt.1).and.(ny.eq.1)) then
       write(*,*) 'Sorry, y-dimension cannot be switched off if '
       write(*,*) '     z-dimension switched on'
       stop 13
    endif
    if(ny.eq.1) then
       diff_idim = 1
    elseif(nz.eq.1) then
       diff_idim = 2
    else
       diff_idim = 3
    endif
    !
    ! The metric
    !          
    diff_imetric = imetric
    !
    ! Switch on/off dimensions
    !
    diff_dimy = .true.
    diff_dimz = .true.
    if(diff_idim.lt.3) diff_dimz=.false.
    if(diff_idim.lt.2) diff_dimy=.false.
    !
    ! Check array sizes
    !
    if(nx.lt.3) then
       write(*,*) 'ERROR: Array size to small in x-direction'
       stop 13
    endif
    if(diff_dimy) then
       if(ny.lt.3) then
          write(*,*) 'ERROR: Array size to small in y-direction'
          stop 13
       endif
    endif
    if(diff_dimz) then
       if(nz.lt.3) then
          write(*,*) 'ERROR: Array size to small in z-direction'
          stop 13
       endif
    endif
    !
    ! First allocate the coordinates
    !
    allocate(diff_x(nx))
    allocate(diff_y(ny))
    allocate(diff_z(nz))
    !
    ! Depending on the dimension, how many bands of the matrix do we need
    !
    if(diff_idim.eq.1) then
       diff_nband = 1
    elseif(diff_idim.eq.2) then
       diff_nband = 3
    elseif(diff_idim.eq.3) then
       diff_nband = 9
    else
       stop 111
    endif
    !
    ! In case of less than 3 dimensions, we must know at which
    ! z (and y, in case of 1-D) the equations should be solved.
    ! Also we wish to know the values of zl,zr (yl,yr) of the
    ! cell boundaries in this(ese) extra dimenion(s)
    !
    if(diff_imetric.eq.METRIC_XYZ) then
       !
       ! Cartesian [X,Y,Z]
       !
       if(.not.diff_dimy) then 
          diff_y(1) = 0.d0
          diff_yl   = -0.5d0
          diff_yr   = 0.5d0
       endif
       if(.not.diff_dimz) then
          diff_z(1) = 0.d0
          diff_zl   = -0.5d0
          diff_zr   = 0.5d0
       endif
    elseif(diff_imetric.eq.METRIC_RTHPHI) then
       !
       ! Spherical [R,Theta,Phi]
       !
       if(.not.diff_dimy) then 
          diff_y(1) = 1.5707963268d0      
          diff_yl   = 0.d0
          diff_yl   = 3.14159265359d0
       endif
       if(.not.diff_dimz) then
          diff_z(1) = 0.d0
          diff_zl   = -3.14159265359d0
          diff_zr   = 3.14159265359d0
       endif
    elseif(diff_imetric.eq.METRIC_RZPHI) then
       !
       ! Cylindrical [R,Z,Phi]
       !
       if(.not.diff_dimy) then 
          diff_y(1) = 0.d0
          diff_yl   = -0.5d0
          diff_yr   =  0.5d0
       endif
       if(.not.diff_dimz) then
          diff_z(1) = 0.d0
          diff_zl   = -3.14159265359d0
          diff_zr   = 3.14159265359d0
       endif
    else
       write(*,*) 'Sorry, metric type ',diff_imetric,' not known.'
       stop 13
    endif
    !
    ! Allocate arrays
    !
    allocate(diff_wxx(nx,ny,nz))
    allocate(diff_wxy(nx,ny,nz))
    allocate(diff_wyy(nx,ny,nz))
    allocate(diff_wxz(nx,ny,nz))
    allocate(diff_wyz(nx,ny,nz))
    allocate(diff_pxl(ny,nz))
    allocate(diff_qxl(ny,nz))
    allocate(diff_rxl(ny,nz))
    allocate(diff_pxr(ny,nz))
    allocate(diff_qxr(ny,nz))
    allocate(diff_rxr(ny,nz))
    allocate(diff_pyl(nx,nz))
    allocate(diff_qyl(nx,nz))
    allocate(diff_ryl(nx,nz))
    allocate(diff_pyr(nx,nz))
    allocate(diff_qyr(nx,nz))
    allocate(diff_ryr(nx,nz))
    allocate(diff_pzl(nx,ny))
    allocate(diff_qzl(nx,ny))
    allocate(diff_rzl(nx,ny))
    allocate(diff_pzr(nx,ny))
    allocate(diff_qzr(nx,ny))
    allocate(diff_rzr(nx,ny))
    allocate(diff_s(nx,ny,nz))
    allocate(diff_ax(nx,ny,nz))
    allocate(diff_ay(nx,ny,nz))
    allocate(diff_az(nx,ny,nz))
    allocate(diff_b(nx,ny,nz))
    allocate(diff_j(nx,ny,nz))
    allocate(diff_hx(nx,ny,nz))
    allocate(diff_hy(nx,ny,nz))
    allocate(diff_hz(nx,ny,nz))
    allocate(diff_error(nx,ny,nz))
    allocate(diff_fxx(nx,ny,nz))
    allocate(diff_fyy(nx,ny,nz))
    allocate(diff_fzz(nx,ny,nz))
    allocate(diff_fxy(nx,ny,nz))
    allocate(diff_fxz(nx,ny,nz))
    allocate(diff_fyz(nx,ny,nz))
    allocate(diff_tgas(nx,ny,nz))
    allocate(diff_t_irrad(nx,ny,nz))
    allocate(diff_acp(NBMAX,ZBMAX,diff_lda))
    allocate(diff_rhs(diff_lda))
    allocate(diff_sol(diff_lda))
    allocate(diff_mixb(NBMAX))
    !
    ! If time-dependent, then allocate more arrays
    !
    if(diff_timedep) then
       allocate(diff_cvrho(nx,ny,nz))
       allocate(diff_gfact(nx,ny,nz))
    endif
    !
    ! Reset the input arrays to zero (to avoid accidential crazy values)
    !
    diff_j(:,:,:)         = 0.d0
    diff_tgas(:,:,:)      = 0.d0
    diff_t_irrad(:,:,:)   = 0.d0
    diff_wxx(:,:,:)       = 0.d0
    diff_wyy(:,:,:)       = 0.d0
    diff_wxy(:,:,:)       = 0.d0
    diff_wxz(:,:,:)       = 0.d0
    diff_wyz(:,:,:)       = 0.d0
    diff_s(:,:,:)         = 0.d0
    diff_ax(:,:,:)        = 0.d0
    diff_ay(:,:,:)        = 0.d0
    diff_az(:,:,:)        = 0.d0
    diff_b(:,:,:)         = 0.d0
    if(diff_timedep) then
       diff_cvrho(:,:,:)  = 0.d0
       diff_gfact(:,:,:)  = 0.d0
    endif
    diff_pxl(:,:)         = 0.d0
    diff_qxl(:,:)         = 0.d0
    diff_rxl(:,:)         = 0.d0
    diff_pxr(:,:)         = 0.d0
    diff_qxr(:,:)         = 0.d0
    diff_rxr(:,:)         = 0.d0
    diff_pyl(:,:)         = 0.d0
    diff_qyl(:,:)         = 0.d0
    diff_ryl(:,:)         = 0.d0
    diff_pyr(:,:)         = 0.d0
    diff_qyr(:,:)         = 0.d0
    diff_ryr(:,:)         = 0.d0
    diff_pzl(:,:)         = 0.d0
    diff_qzl(:,:)         = 0.d0
    diff_rzl(:,:)         = 0.d0
    diff_pzr(:,:)         = 0.d0
    diff_qzr(:,:)         = 0.d0
    diff_rzr(:,:)         = 0.d0
    !
  end subroutine diff_init


  !---------------------------------------------------------------
  !                          FINISH
  !---------------------------------------------------------------
  subroutine diff_finish()
    implicit none
    if(allocated(diff_x))       deallocate(diff_x)
    if(allocated(diff_y))       deallocate(diff_y)
    if(allocated(diff_z))       deallocate(diff_z)
    if(allocated(diff_wxx))     deallocate(diff_wxx)
    if(allocated(diff_wxy))     deallocate(diff_wxy)
    if(allocated(diff_wyy))     deallocate(diff_wyy)
    if(allocated(diff_wxz))     deallocate(diff_wxz)
    if(allocated(diff_wyz))     deallocate(diff_wyz)
    if(allocated(diff_pxl))     deallocate(diff_pxl)
    if(allocated(diff_qxl))     deallocate(diff_qxl)
    if(allocated(diff_rxl))     deallocate(diff_rxl)
    if(allocated(diff_pxr))     deallocate(diff_pxr)
    if(allocated(diff_qxr))     deallocate(diff_qxr)
    if(allocated(diff_rxr))     deallocate(diff_rxr)
    if(allocated(diff_pyl))     deallocate(diff_pyl)
    if(allocated(diff_qyl))     deallocate(diff_qyl)
    if(allocated(diff_ryl))     deallocate(diff_ryl)
    if(allocated(diff_pyr))     deallocate(diff_pyr)
    if(allocated(diff_qyr))     deallocate(diff_qyr)
    if(allocated(diff_ryr))     deallocate(diff_ryr)
    if(allocated(diff_pzl))     deallocate(diff_pzl)
    if(allocated(diff_qzl))     deallocate(diff_qzl)
    if(allocated(diff_rzl))     deallocate(diff_rzl)
    if(allocated(diff_pzr))     deallocate(diff_pzr)
    if(allocated(diff_qzr))     deallocate(diff_qzr)
    if(allocated(diff_rzr))     deallocate(diff_rzr)
    if(allocated(diff_s))       deallocate(diff_s)
    if(allocated(diff_ax))      deallocate(diff_ax)
    if(allocated(diff_ay))      deallocate(diff_ay)
    if(allocated(diff_az))      deallocate(diff_az)
    if(allocated(diff_b))       deallocate(diff_b)
    if(allocated(diff_j))       deallocate(diff_j)
    if(allocated(diff_hx))      deallocate(diff_hx)
    if(allocated(diff_hy))      deallocate(diff_hy)
    if(allocated(diff_hz))      deallocate(diff_hz)
    if(allocated(diff_error))   deallocate(diff_error)
    if(allocated(diff_fxx))     deallocate(diff_fxx)
    if(allocated(diff_fyy))     deallocate(diff_fyy)
    if(allocated(diff_fzz))     deallocate(diff_fzz)
    if(allocated(diff_fxy))     deallocate(diff_fxy)
    if(allocated(diff_fxz))     deallocate(diff_fxz)
    if(allocated(diff_fyz))     deallocate(diff_fyz)
    if(allocated(diff_acp))     deallocate(diff_acp)
    if(allocated(diff_rhs))     deallocate(diff_rhs)
    if(allocated(diff_sol))     deallocate(diff_sol)
    if(allocated(diff_mixb))    deallocate(diff_mixb)
    if(allocated(diff_cvrho))   deallocate(diff_cvrho)
    if(allocated(diff_t_irrad)) deallocate(diff_t_irrad)
    if(allocated(diff_tgas))    deallocate(diff_tgas)
    if(allocated(diff_gfact))   deallocate(diff_gfact)
  end subroutine diff_finish

  
  !---------------------------------------------------------------
  !                  THE DIFFUSION MAIN ROUTINE
  !
  ! First call diff_init() to allocate all arrays and set things
  ! up. Then insert your problem values into the following
  ! arrays (always with prefix diff_, see above):
  !
  !    x,y,z         The grid in x, y and z (for 2-D or 1-D the
  !                  z resp y,z arrays are ignored).
  !    wxx etc       The dimensionless traceless Eddington factors.
  !    pxl,qxl,rxl   The p,q,r of the boundary condition in 
  !                    the x-dimension, at ix=1.
  !    pxr,qxr,rxr   The p,q,r of the boundary condition in 
  !                    the x-dimension, at ix=nx.
  !    pyl,qyl,ryl   The p,q,r of the boundary condition in 
  !                    the y-dimension, at iy=1.
  !     .... etc
  !    s             The source term of the equation 
  !                    (Note: this is for fluxes H=F/4pi, so this 
  !                     source is s=q/4pi, where q is erg/s/cm^3)
  !    ax            The alpha in x-direction
  !    ay            The alpha in y-direction
  !    az            The alpha in z-direction
  !    b             The beta factor before J on the rhs of the eq
  !    j             The initial guess for the mean intensity J
  !    tgas          (if usetgas is .true.) then use this as
  !                  initial guess instead of j.
  !
  ! Then call this subroutine.
  !
  ! ARGUMENTS:
  !
  !    convcrit      The convergence criterion (error tolerance)
  !    itermax       The maximum number of iterations
  !    savematrhs    If .true. then write the band matrix and rhs
  !                  to file (useful for debugging purposes).
  !    timestep      (for time-dependence) Time step
  !    usetgas       If .true.: Convert from diff_tgas to diff_j 
  !                  as initial guess, and convert from diff_j
  !                  back to diff_tgas as the result. Default
  !                  is .false. (i.e. solve for diff_j). 
  !
  ! RESULT:
  !    j             The result for the mean intensity J
  !    hx            The flux in x-direction between ix and ix+1
  !    hy            The flux in y-direction between iy and iy+1
  !    hz            The flux in z-direction between iz and iz+1
  !    error         The error in the flux conservation at ix,iy,iz.
  !
  ! NOTE: The returned error is the error in the flux conservation.
  !       Sometimes it can happen that a solution is found (i.e. a 
  !       small error for linbcg.f), but still with a large error
  !       in the flux conservation. For instance, when the density
  !       becomes very low (optically thin), then the equations 
  !       become too stiff, and this error can happen. One should
  !       solve this by making a lower limit to the density. After
  !       all: optically thin is optically thin. 
  !
  ! NOTE: nz.gt.1 while ny.eq.1 is not allowed. I.e. 2-D can only be
  !       done with nz.eq.1, and 1-D only by nz.eq.1 and ny.eq.1.
  !
  ! NOTE: wzz is not an argument, because it follows from wxx and
  !       wyy by: wzz = - wxx - wyy. Similarly, in case of 1-D
  !       diffusion equation, wyy = - 0.5 wxx and wzz = - 0.5 wxx.
  !
  ! NOTE: In 2-D one can pass dummy 1-element arrays for the 
  !       following arguments: wxz,wyz,pzl,qzl,rzl,pzr,qzr,rzr.
  !---------------------------------------------------------------
  subroutine solve_diffusion(convcrit,itermax,savematrhs,timestep,usetgas)
    implicit none
    integer :: itermax
    doubleprecision :: convcrit
    doubleprecision :: comp(-1:1,-1:1,-1:1)
    doubleprecision :: err,dum,dum1,dum2
    doubleprecision :: dumxl,dumxr,dumyl,dumyr,dumzl,dumzr
    doubleprecision :: vol,surf_m,surf_p,norm
    doubleprecision :: eta,xx,yy,zz
    doubleprecision, optional :: timestep
    integer :: ix,iy,iz,iter,im,ixb,ixe,iyb,iye,izb,ize,iix,iiy,iiz
    integer :: nxb,nxe,nyb,nye,nzb,nze
    logical :: iwarn_fxybc,savematrhs
    logical, optional :: usetgas
    logical :: use_tgas
    !
    ! Interpret usetgas
    !
    use_tgas = .false.
    if(present(usetgas)) then
       use_tgas = usetgas
    endif
    !
    ! Check the grid for monotonicity
    !
    if(diff_x(2).gt.diff_x(1)) then
       diff_sgnx = 1.d0
    else
       diff_sgnx = -1.d0
    endif
    do ix=3,diff_nx
       if(diff_sgnx*(diff_x(ix)-diff_x(ix-1)).lt.0.d0) then
          write(*,*) 'X-grid not monotonic!'
          stop 13
       endif
    enddo
    if(diff_dimy) then
       if(diff_y(2).gt.diff_y(1)) then
          diff_sgny = 1.d0
       else
          diff_sgny = -1.d0
       endif
       do iy=3,diff_ny
          if(diff_sgny*(diff_y(iy)-diff_y(iy-1)).lt.0.d0) then
             write(*,*) 'Y-grid not monotonic!'
             stop 13
          endif
       enddo
    else
       diff_sgny = 1.d0
    endif
    if(diff_dimz) then
       if(diff_z(2).gt.diff_z(1)) then
          diff_sgnz = 1.d0
       else
          diff_sgnz = -1.d0
       endif
       do iz=3,diff_nz
          if(diff_sgnz*(diff_z(iz)-diff_z(iz-1)).lt.0.d0) then
             write(*,*) 'Z-grid not monotonic!'
             stop 13
          endif
       enddo
    else
       diff_sgnz = 1.d0
    endif
    !
    ! If usetgas, then convert from diff_tgas to diff_j
    !
    if(use_tgas) then
       diff_j(:,:,:) = ( diff_tgas(:,:,:)**4 -                    &
            ( diff_t_irrad(:,:,:) + diff_tmin )**4 ) / pisigsb
    endif
    !
    ! Check source, a and b
    !     
    do iz=1,diff_nz
       do iy=1,diff_ny
          do ix=1,diff_nx
             if(diff_s(ix,iy,iz).lt.0.d0) then
                write(*,*) 'ERROR in calling solve_diffusion():'
                write(*,*) '      Source contains negative'
                write(*,*) '      numbers'
                stop 13
             endif
             if(diff_b(ix,iy,iz).lt.0.d0) then
                write(*,*) 'ERROR in calling solve_diffusion():'
                write(*,*) '      Opacity b contains negative'
                write(*,*) '      numbers'
                stop 13
             endif
          enddo
       enddo
    enddo
    !
    ! Create the non-traceless, covariant Eddington tensor components
    ! f^ij out of the traceless local (orthogonal) Eddington tensor
    ! components w^ij.
    !
    ! NOTE: In the following it is assumed that if diff_dimz is true, then
    !       diff_dimy is also true. Or in other words, if diff_dimy is untrue, then
    !       we are in 1-D mode.
    !     
    do iz=1,diff_nz
       do iy=1,diff_ny
          do ix=1,diff_nx
             xx    = diff_x(ix)
             yy    = diff_y(iy)
             zz    = diff_z(iz)
             diff_fxx(ix,iy,iz) = ( diff_wxx(ix,iy,iz) + (1.d0/3.d0) )          &
                           / sqroot_gij(diff_imetric,1,xx,yy,zz)**2
             if(diff_dimy) then
                diff_fxy(ix,iy,iz) = diff_wxy(ix,iy,iz)                         &
                           / sqroot_gij(diff_imetric,1,xx,yy,zz)                &
                           / sqroot_gij(diff_imetric,2,xx,yy,zz)
                diff_fyy(ix,iy,iz) = ( diff_wyy(ix,iy,iz) + (1.d0/3.d0) )       &
                           / sqroot_gij(diff_imetric,2,xx,yy,zz)**2
             else
                diff_fxy(ix,iy,iz) = 0.d0
                diff_fyy(ix,iy,iz) = ( (1.d0/3.d0) - 0.5d0*diff_wxx(ix,iy,iz) ) &
                           / sqroot_gij(diff_imetric,2,xx,yy,zz)**2
             endif
             if(diff_dimz) then
                diff_fxz(ix,iy,iz) = diff_wxz(ix,iy,iz)                         &
                           / sqroot_gij(diff_imetric,1,xx,yy,zz  )              &
                           / sqroot_gij(diff_imetric,3,xx,yy,zz)
                diff_fyz(ix,iy,iz) = diff_wyz(ix,iy,iz)                         &
                           / sqroot_gij(diff_imetric,2,xx,yy,zz)                &
                           / sqroot_gij(diff_imetric,3,xx,yy,zz)
             else
                diff_fxz(ix,iy,iz) = 0.d0
                diff_fyz(ix,iy,iz) = 0.d0
             endif
             if(diff_dimy) then
                diff_fzz(ix,iy,iz) = ( (1.d0/3.d0) -                            &
                           diff_wxx(ix,iy,iz) - diff_wyy(ix,iy,iz) )            &
                           / sqroot_gij(diff_imetric,3,xx,yy,zz)**2
             else
                diff_fzz(ix,iy,iz) = ( (1.d0/3.d0) - 0.5d0 * diff_wxx(ix,iy,iz) ) &
                           / sqroot_gij(diff_imetric,3,xx,yy,zz)**2
             endif
          enddo
       enddo
    enddo
    !
    ! Check whether at the boundaries the fxy,fxz,fyz are non-zero.
    ! If so, then set to zero, and create a warning, because with
    ! non-zero fxy,fxz,fyz the boundary conditions become rather
    ! fishy at the ribbons and corners of the 3-D domain. 
    !
    iwarn_fxybc = .false.
    !
    do iz=1,diff_nz
       do iy=1,diff_ny
          ix=1
          if((diff_fxy(ix,iy,iz).ne.0.d0).or.         &
             (diff_fxz(ix,iy,iz).ne.0.d0).or.         &
             (diff_fyz(ix,iy,iz).ne.0.d0)) then
             iwarn_fxybc = .true.
          endif
          diff_fxy(ix,iy,iz)=0.d0
          diff_fxz(ix,iy,iz)=0.d0
          diff_fyz(ix,iy,iz)=0.d0
          ix=diff_nx
          if((diff_fxy(ix,iy,iz).ne.0.d0).or.         &
             (diff_fxz(ix,iy,iz).ne.0.d0).or.         &
             (diff_fyz(ix,iy,iz).ne.0.d0)) then
             iwarn_fxybc = .true.
          endif
          diff_fxy(ix,iy,iz)=0.d0
          diff_fxz(ix,iy,iz)=0.d0
          diff_fyz(ix,iy,iz)=0.d0
       enddo
    enddo
    if(diff_dimy) then
       do iz=1,diff_nz
          do ix=1,diff_nx
             iy=1
             if((diff_fxy(ix,iy,iz).ne.0.d0).or.      &
                (diff_fxz(ix,iy,iz).ne.0.d0).or.      &
                (diff_fyz(ix,iy,iz).ne.0.d0)) then
                iwarn_fxybc = .true.
             endif
             diff_fxy(ix,iy,iz)=0.d0
             diff_fxz(ix,iy,iz)=0.d0
             diff_fyz(ix,iy,iz)=0.d0
             iy=diff_ny
             if((diff_fxy(ix,iy,iz).ne.0.d0).or.      &
                (diff_fxz(ix,iy,iz).ne.0.d0).or.      &
                (diff_fyz(ix,iy,iz).ne.0.d0)) then
                iwarn_fxybc = .true.
             endif
             diff_fxy(ix,iy,iz)=0.d0
             diff_fxz(ix,iy,iz)=0.d0
             diff_fyz(ix,iy,iz)=0.d0
          enddo
       enddo
    endif
    if(diff_dimz) then
       do ix=1,diff_nx
          do iy=1,diff_ny
             iz=1
             if((diff_fxy(ix,iy,iz).ne.0.d0).or.      &
                (diff_fxz(ix,iy,iz).ne.0.d0).or.      &
                (diff_fyz(ix,iy,iz).ne.0.d0)) then
                iwarn_fxybc = .true.
             endif
             diff_fxy(ix,iy,iz)=0.d0
             diff_fxz(ix,iy,iz)=0.d0
             diff_fyz(ix,iy,iz)=0.d0
             iz=diff_nz
             if((diff_fxy(ix,iy,iz).ne.0.d0).or.      &
                (diff_fxz(ix,iy,iz).ne.0.d0).or.      &
                (diff_fyz(ix,iy,iz).ne.0.d0)) then
                iwarn_fxybc = .true.
             endif
             diff_fxy(ix,iy,iz)=0.d0
             diff_fxz(ix,iy,iz)=0.d0
             diff_fyz(ix,iy,iz)=0.d0
          enddo
       enddo
    endif
    if(iwarn_fxybc) then
       write(*,*) 'WARNING in solve_diffusion():'
       write(*,*) '  Skew Eddington factors were non-zero at '
       write(*,*) '  boundaries. They have been set to zero there.'
    endif
    !
    ! Now set up the matrix and the rhs
    !
    call create_the_band_matrix_and_diff_rhs()
    !
    ! If time-dependent, then add some terms
    !
    if(diff_timedep.and.(present(timestep))) then
       call add_time_dependence_terms(timestep)
    endif
    !
    ! For debugging: write matrix and the rhs. 
    !     
    if(savematrhs) then
       call write_band_matrix(diff_lda,diff_nband,ZBMAX)
       call write_righthandside(diff_lda)
    endif
    !
    ! Next solve the equations
    !
    call linbcg(diff_lda,diff_rhs,diff_sol,ICONVCRIT,convcrit,itermax,iter,err)
    !
    ! Store convergence information
    !
    diff_mateq_error = err
    diff_mateq_iter  = iter
    !
    ! Messages
    !
    if(.not.diff_silent) then
       if(iter.lt.itermax) then
          write(*,*) ' Convergence after ',iter,' iterations'
          write(*,*) ' Error  :',err
       else
          write(*,*) ' No convergence after ',iter,' iterations'
          write(*,*) ' Error  :',err
       endif
    endif
    !
    ! Copy things back into the arrays
    !     
    do iz=1,diff_nz
       do iy=1,diff_ny
          do ix=1,diff_nx
             !
             ! First get the eta-function
             !
             xx    = diff_x(ix)
             yy    = diff_y(iy)
             zz    = diff_z(iz)
             call eta_function(diff_imetric,xx,yy,zz,eta)
             !
             ! Then retrieve the solution
             !
             im = (ix-1) + (iy-1)*diff_nx + (iz-1)*diff_nx*diff_ny + 1
             diff_j(ix,iy,iz) = diff_sol(im) / eta
             !
          enddo
       enddo
    enddo
    !
    ! If requested, compute 
    !
    if(use_tgas) then
       diff_tgas(:,:,:) = ( diff_j(:,:,:)*pisigsb +                         &
            ( diff_t_irrad(:,:,:) + diff_tmin )**4 )**0.25
    endif
    !
    ! Recreate the fluxes from the solution
    !
    ! ...First set up stuff
    !
    ixb=-1
    ixe=+1
    nxb=2
    nxe=diff_nx-1
    if(diff_dimy) then
       nyb=2
       nye=diff_ny-1
       iyb=-1
       iye=+1
    else
       nyb=1
       nye=1
       iyb=0
       iye=0
    endif
    if(diff_dimz) then
       nzb=2
       nze=diff_nz-1
       izb=-1
       ize=+1
    else
       nzb=1
       nze=1
       izb=0
       ize=0
    endif
    !
    ! ...then clear arrays
    !
    do iz=1,diff_nz
       do iy=1,diff_ny
          do ix=1,diff_nx
             diff_error(ix,iy,iz) = 0.d0
             diff_hx(ix,iy,iz) = 0.d0
             if(diff_dimy) diff_hy(ix,iy,iz) = 0.d0
             if(diff_dimz) diff_hz(ix,iy,iz) = 0.d0
          enddo
       enddo
    enddo
    !
    ! ...the Hx flux
    !
    do iz=nzb,nze
       do iy=nyb,nye
          do ix=1,diff_nx-1
             call find_components_hflux(1,1,ix,iy,iz,comp)
             dum = 0.d0
             do iiz=izb,ize
                do iiy=iyb,iye
                   do iix=0,1
                      im = (ix+iix-1) + (iy+iiy-1)*diff_nx + (iz+iiz-1)*diff_nx*diff_ny + 1
                      dum=dum+comp(iix,iiy,iiz)*diff_sol(im)
                   enddo
                enddo
             enddo
             diff_hx(ix,iy,iz)=dum
          enddo
       enddo
    enddo
    !
    ! ...the Hy flux
    !
    if(diff_dimy) then
    do iz=nzb,nze
       do iy=1,diff_ny-1
          do ix=nxb,nxe
             call find_components_hflux(2,1,ix,iy,iz,comp)
             dum = 0.d0
             do iiz=izb,ize
                do iiy=0,1
                   do iix=ixb,ixe
                      im = (ix+iix-1) + (iy+iiy-1)*diff_nx + (iz+iiz-1)*diff_nx*diff_ny + 1
                      dum=dum+comp(iix,iiy,iiz)*diff_sol(im)
                   enddo
                enddo
             enddo
             diff_hy(ix,iy,iz)=dum
          enddo
       enddo
    enddo
    endif
    !
    ! ...the Hz flux
    !
    if(diff_dimz) then
    do iz=1,diff_nz-1
       do iy=nyb,nye
          do ix=nxb,nxe
             call find_components_hflux(3,1,ix,iy,iz,comp)
             dum = 0.d0
             do iiz=0,1
                do iiy=iyb,iye
                   do iix=ixb,ixe
                      im = (ix+iix-1) + (iy+iiy-1)*diff_nx + (iz+iiz-1)*diff_nx*diff_ny + 1
                      dum=dum+comp(iix,iiy,iiz)*diff_sol(im)
                   enddo
                enddo
             enddo
             diff_hz(ix,iy,iz)=dum
          enddo
       enddo
    enddo
    endif
    !
    ! Now check whether the conservation equation is indeed satisfied
    !
    if(.not.diff_timedep) then
       ixb=2
       ixe=diff_nx-1
       if(diff_dimy) then 
          iyb=2
          iye=diff_ny-1
       else
          iyb=1
          iye=1
       endif
       if(diff_dimz) then
          izb=2
          ize=diff_nz-1
       else
          izb=1
          ize=1
       endif
       do iz=izb,ize
          do iy=iyb,iye
             do ix=ixb,ixe
                dum    = 0.d0
                norm   = 0.d0
                surf_p = get_surface_element(1,2*ix+1,2*iy,2*iz)
                surf_m = get_surface_element(1,2*ix-1,2*iy,2*iz)
                dum    = surf_p * diff_sgnx * diff_hx(ix,iy,iz)          &
                       - surf_m * diff_sgnx * diff_hx(ix-1,iy,iz)
                norm   = abs(surf_p * diff_hx(ix,iy,iz)) +               &
                         abs(surf_m * diff_hx(ix-1,iy,iz))
                if(diff_dimy) then
                    surf_p = get_surface_element(2,2*ix,2*iy+1,2*iz)
                    surf_m = get_surface_element(2,2*ix,2*iy-1,2*iz)
                    dum    = dum + surf_p * diff_sgny * diff_hy(ix,iy,iz)  &
                                 - surf_m * diff_sgny * diff_hy(ix,iy-1,iz)
                    norm   = norm + abs(surf_p * diff_hy(ix,iy,iz)) +      &
                                    abs(surf_m * diff_hy(ix,iy-1,iz))
                endif
                if(diff_dimz) then
                    surf_p = get_surface_element(3,2*ix,2*iy,2*iz+1)
                    surf_m = get_surface_element(3,2*ix,2*iy,2*iz-1)
                    dum    = dum + surf_p * diff_sgnz * diff_hz(ix,iy,iz)  &
                                 - surf_m * diff_sgnz * diff_hz(ix,iy,iz-1)
                    norm   = norm + abs(surf_p * diff_hz(ix,iy,iz)) +      &
                                    abs(surf_m * diff_hz(ix,iy,iz-1))
                endif
                vol    = get_volume_element(ix,iy,iz)
                dum    = dum - vol * ( diff_s(ix,iy,iz) - diff_b(ix,iy,iz) * diff_j(ix,iy,iz) )
                diff_error(ix,iy,iz) = abs(dum) / norm
             enddo
          enddo
       enddo
       !
       ! Now check the global conservation of energy
       !
       ! First the total emission - absorption
       !
       dum1    = 0.d0
       do iz=izb,ize
          do iy=iyb,iye
             do ix=ixb,ixe
                vol    = get_volume_element(ix,iy,iz)
                dum1   = dum1 + vol * ( diff_s(ix,iy,iz) - diff_b(ix,iy,iz) * diff_j(ix,iy,iz) )
             enddo
          enddo
       enddo
       dum1 = dum1 * 12.5663706144d0
       !
       ! Then the flux through the boundaries
       !
       dumxl = 0.d0
       dumxr = 0.d0
       dumyl = 0.d0
       dumyr = 0.d0
       dumzl = 0.d0
       dumzr = 0.d0
       norm  = 0.d0
       do iz=izb,ize
          do iy=iyb,iye
             surf_p = get_surface_element(1,2*diff_nx-1,2*iy,2*iz)
             surf_m = get_surface_element(1,3,2*iy,2*iz)
             dumxr  = dumxr + surf_p * diff_sgnx * diff_hx(diff_nx-1,iy,iz)
             dumxl  = dumxl - surf_m * diff_sgnx * diff_hx(1,iy,iz)
             norm   = norm + abs(surf_p*diff_hx(diff_nx-1,iy,iz))                &
                           + abs(surf_m*diff_hx(1,iy,iz))
          enddo
       enddo
       if(diff_dimy) then
       do iz=izb,ize
          do ix=ixb,ixe
             surf_p = get_surface_element(2,2*ix,2*diff_ny-1,2*iz)
             surf_m = get_surface_element(2,2*ix,3,2*iz)
             dumyr  = dumyr + surf_p * diff_sgny * diff_hy(ix,diff_ny-1,iz)
             dumyl  = dumyl - surf_m * diff_sgny * diff_hy(ix,1,iz)
             norm   = norm + abs(surf_p*diff_hy(ix,diff_ny-1,iz))           &
                           + abs(surf_m*diff_hy(ix,1,iz))
          enddo
       enddo
       endif
       if(diff_dimz) then
       do iy=iyb,iye
          do ix=ixb,ixe
             surf_p = get_surface_element(3,2*ix,2*iy,2*diff_nz-1)
             surf_m = get_surface_element(3,2*ix,2*iy,3)
             dumzr  = dumzr + surf_p * diff_sgnz * diff_hz(ix,iy,diff_nz-1)
             dumzl  = dumzl - surf_m * diff_sgnz * diff_hz(ix,iy,1)
             norm   = norm + abs(surf_p*diff_hz(ix,iy,diff_nz-1))           &
                           + abs(surf_m*diff_hz(ix,iy,1))
          enddo
       enddo
       endif
       dumxl = dumxl * 12.5663706144d0
       dumxr = dumxr * 12.5663706144d0
       dumyl = dumyl * 12.5663706144d0
       dumyr = dumyr * 12.5663706144d0
       dumzl = dumzl * 12.5663706144d0
       dumzr = dumzr * 12.5663706144d0
       norm  = norm  * 12.5663706144d0
       dum2 = dumxr+dumxl+dumyr+dumyl+dumzr+dumzl
       diff_fluxcons_error = 2.d0*abs(dum2-dum1)/norm
       if(.not.diff_silent) then
          write(*,*) 'Flux out X_in             = ',dumxl
          write(*,*) 'Flux out X_out            = ',dumxr
          if(diff_dimy) then
             write(*,*) 'Flux out Y_in             = ',dumyl
             write(*,*) 'Flux out Y_out            = ',dumyr
          endif
          if(diff_dimz) then
             write(*,*) 'Flux out Z_in             = ',dumzl
             write(*,*) 'Flux out Z_out            = ',dumzr
          endif
          write(*,*) '--------------------------------------------'
          write(*,*) 'Flux out total            = ',dum2
          write(*,*) 'Source total              = ',dum1
          write(*,*) '--------------------------------------------'
          write(*,*) 'Error in total flux cons  = ',diff_fluxcons_error
       endif
    endif
    !
    ! Done...
    !     
  end subroutine solve_diffusion


  !--------------------------------------------------------------
  !                    SET UP THE MAIN BAND MATRIX
  !--------------------------------------------------------------
  subroutine create_the_band_matrix_and_diff_rhs()
    implicit none
    doubleprecision :: matelem(-1:1,-1:1,-1:1)
    doubleprecision :: p,eta,x,y,z
    integer :: im,izm,ibm
    integer :: ix,iy,iz
    integer :: ixb,ixe,iyb,iye,izb,ize
    !
    ! Then clear the matrix and boundary condition arrays
    !
    do im=1,diff_lda
       do izm=1,ZBMAX
          do ibm=1,NBMAX
             diff_acp(ibm,izm,im) = 0.d0
          enddo
       enddo
    enddo
    !
    ! Set the offset positions of the bands (measured horizontally)
    !
    diff_mixb(1) = 0                               ! The diagonal and x-bands, mid band in z
    if(diff_idim.ge.2) then
       diff_mixb(2) = - diff_nx                    ! The lower band in y-dir,  mid band in z
       diff_mixb(3) = + diff_nx                    ! The upper band in y-dir,  mid band in z
    endif
    if(diff_idim.eq.3) then
       diff_mixb(4) = 0 - diff_nx*diff_ny          ! The diagonal and x-bands, lower band in z
       diff_mixb(5) = - diff_nx - diff_nx*diff_ny  ! The lower band in y-dir,  lower band in z
       diff_mixb(6) = + diff_nx - diff_nx*diff_ny  ! The upper band in y-dir,  lower band in z
       diff_mixb(7) = 0 + diff_nx*diff_ny          ! The diagonal and x-bands, upper band in z
       diff_mixb(8) = - diff_nx + diff_nx*diff_ny  ! The lower band in y-dir,  upper band in z
       diff_mixb(9) = + diff_nx + diff_nx*diff_ny  ! The upper band in y-dir,  upper band in z
    endif
    !
    ! Now find out the boundaries within which to fill the matrix with
    ! the PDE. If a dimension is active, then we must skip the boundary
    ! cells. If not, then use index 1 simply. 
    !
    ixb = 2
    ixe = diff_nx - 1
    if(diff_dimy) then
       iyb = 2
       iye = diff_ny - 1
    else
       iyb = 1
       iye = 1
    endif
    if(diff_dimz) then
       izb = 2
       ize = diff_nz - 1
    else
       izb = 1
       ize = 1
    endif
    !
    ! Now a loop over this domain to fill the matrix with the PDE
    !
    do iz=izb,ize
       do iy=iyb,iye
          do ix=ixb,ixe
             !
             ! Find the (local) matrix elements
             !
             call find_matrix_elements_pde(ix,iy,iz,matelem)
             !
             ! Plug them into the big matrix
             !
             im = (ix-1) + (iy-1)*diff_nx + (iz-1)*diff_nx*diff_ny + 1
             !  
             diff_acp(1,2,im) = diff_acp(1,2,im) + matelem( 0, 0, 0)
             diff_acp(1,1,im) = diff_acp(1,1,im) + matelem(-1, 0, 0)
             diff_acp(1,3,im) = diff_acp(1,3,im) + matelem(+1, 0, 0)
             if(diff_dimy) then
                diff_acp(2,2,im) = diff_acp(2,2,im) + matelem( 0,-1, 0)
                diff_acp(2,1,im) = diff_acp(2,1,im) + matelem(-1,-1, 0)
                diff_acp(2,3,im) = diff_acp(2,3,im) + matelem(+1,-1, 0)
                diff_acp(3,2,im) = diff_acp(3,2,im) + matelem( 0,+1, 0)
                diff_acp(3,1,im) = diff_acp(3,1,im) + matelem(-1,+1, 0)
                diff_acp(3,3,im) = diff_acp(3,3,im) + matelem(+1,+1, 0)
             endif
             if(diff_dimz) then
                diff_acp(4,2,im) = diff_acp(4,2,im) + matelem( 0, 0,-1)
                diff_acp(4,1,im) = diff_acp(4,1,im) + matelem(-1, 0,-1)
                diff_acp(4,3,im) = diff_acp(4,3,im) + matelem(+1, 0,-1)
                diff_acp(5,2,im) = diff_acp(5,2,im) + matelem( 0,-1,-1)
                diff_acp(5,1,im) = diff_acp(5,1,im) + matelem(-1,-1,-1)
                diff_acp(5,3,im) = diff_acp(5,3,im) + matelem(+1,-1,-1)
                diff_acp(6,2,im) = diff_acp(6,2,im) + matelem( 0,+1,-1)
                diff_acp(6,1,im) = diff_acp(6,1,im) + matelem(-1,+1,-1)
                diff_acp(6,3,im) = diff_acp(6,3,im) + matelem(+1,+1,-1)
                diff_acp(7,2,im) = diff_acp(7,2,im) + matelem( 0, 0,+1)
                diff_acp(7,1,im) = diff_acp(7,1,im) + matelem(-1, 0,+1)
                diff_acp(7,3,im) = diff_acp(7,3,im) + matelem(+1, 0,+1)
                diff_acp(8,2,im) = diff_acp(8,2,im) + matelem( 0,-1,+1)
                diff_acp(8,1,im) = diff_acp(8,1,im) + matelem(-1,-1,+1)
                diff_acp(8,3,im) = diff_acp(8,3,im) + matelem(+1,-1,+1)
                diff_acp(9,2,im) = diff_acp(9,2,im) + matelem( 0,+1,+1)
                diff_acp(9,1,im) = diff_acp(9,1,im) + matelem(-1,+1,+1)
                diff_acp(9,3,im) = diff_acp(9,3,im) + matelem(+1,+1,+1)
             endif
          enddo
       enddo
    enddo
    !
    ! Reset rhs, and install initial guess
    !     
    do iz=1,diff_nz
       do iy=1,diff_ny
          do ix=1,diff_nx
             !
             ! First get the eta-function
             !
             x     = diff_x(ix)
             y     = diff_y(iy)
             z     = diff_z(iz)
             call eta_function(diff_imetric,x,y,z,eta)
             !
             ! Then fill the initial guess
             !
             im = (ix-1) + (iy-1)*diff_nx + (iz-1)*diff_nx*diff_ny + 1
             diff_sol(im) = eta * diff_j(ix,iy,iz)
             diff_rhs(im) = 0.d0
          enddo
       enddo
    enddo
    !
    ! Install rhs (only at non-boundary gridpoints)
    !     
    do iz=izb,ize
       do iy=iyb,iye
          do ix=ixb,ixe
             im = (ix-1) + (iy-1)*diff_nx + (iz-1)*diff_nx*diff_ny + 1
             diff_rhs(im) = diff_s(ix,iy,iz)
          enddo
       enddo
    enddo
    !
    ! Now the boundary conditions
    !
    ! ... X boundaries
    !
    do iz=izb,ize
       do iy=iyb,iye
          !
          ! Left BC
          !
          ix = 1
          !
          ! First get the eta-function
          !
          x     = diff_x(ix)
          y     = diff_y(iy)
          z     = diff_z(iz)
          call eta_function(diff_imetric,x,y,z,eta)
          !
          ! Now install the BC
          !
          call find_components_hflux(1,-1,ix+1,iy,iz,matelem)
          im = (ix-1) + (iy-1)*diff_nx + (iz-1)*diff_nx*diff_ny + 1
          p  = diff_pxl(iy,iz)
          diff_acp(1,2,im)= diff_acp(1,2,im) + p*matelem(-1,0,0)
          diff_acp(1,3,im)= diff_acp(1,3,im) + p*matelem( 0,0,0)
          if(diff_dimy) then 
             diff_acp(2,2,im) = diff_acp(2,2,im) + p*matelem(-1,-1, 0)
             diff_acp(2,3,im) = diff_acp(2,3,im) + p*matelem( 0,-1, 0)
             diff_acp(3,2,im) = diff_acp(3,2,im) + p*matelem(-1,+1, 0)
             diff_acp(3,3,im) = diff_acp(3,3,im) + p*matelem( 0,+1, 0)
          endif
          if(diff_dimz) then 
             diff_acp(4,2,im) = diff_acp(4,2,im) + p*matelem(-1, 0,-1)
             diff_acp(4,3,im) = diff_acp(4,3,im) + p*matelem( 0, 0,-1)
             diff_acp(5,2,im) = diff_acp(5,2,im) + p*matelem(-1,-1,-1)
             diff_acp(5,3,im) = diff_acp(5,3,im) + p*matelem( 0,-1,-1)
             diff_acp(6,2,im) = diff_acp(6,2,im) + p*matelem(-1,+1,-1)
             diff_acp(6,3,im) = diff_acp(6,3,im) + p*matelem( 0,+1,-1)
             diff_acp(7,2,im) = diff_acp(7,2,im) + p*matelem(-1, 0,+1)
             diff_acp(7,3,im) = diff_acp(7,3,im) + p*matelem( 0, 0,+1)
             diff_acp(8,2,im) = diff_acp(8,2,im) + p*matelem(-1,-1,+1)
             diff_acp(8,3,im) = diff_acp(8,3,im) + p*matelem( 0,-1,+1)
             diff_acp(9,2,im) = diff_acp(9,2,im) + p*matelem(-1,+1,+1)
             diff_acp(9,3,im) = diff_acp(9,3,im) + p*matelem( 0,+1,+1)
          endif
          diff_acp(1,2,im)=diff_acp(1,2,im)+diff_qxl(iy,iz)/eta
          diff_rhs(im)    =diff_rhs(im)    +diff_rxl(iy,iz)
          !
          ! Right BC
          !     
          ix = diff_nx
          !
          ! First get the eta-function
          !
          x     = diff_x(ix)
          y     = diff_y(iy)
          z     = diff_z(iz)
          call eta_function(diff_imetric,x,y,z,eta)
          !
          ! Now install the BC
          !
          call find_components_hflux(1,+1,ix-1,iy,iz,matelem)
          im = (ix-1) + (iy-1)*diff_nx + (iz-1)*diff_nx*diff_ny + 1
          p  = diff_pxr(iy,iz)
          diff_acp(1,2,im)= diff_acp(1,2,im) + p*matelem(+1, 0, 0)
          diff_acp(1,1,im)= diff_acp(1,1,im) + p*matelem( 0, 0, 0)
          if(diff_dimy) then 
             diff_acp(2,2,im) = diff_acp(2,2,im) + p*matelem(+1,-1, 0)
             diff_acp(2,1,im) = diff_acp(2,1,im) + p*matelem( 0,-1, 0)
             diff_acp(3,2,im) = diff_acp(3,2,im) + p*matelem(+1,+1, 0)
             diff_acp(3,1,im) = diff_acp(3,1,im) + p*matelem( 0,+1, 0)
          endif
          if(diff_dimz) then 
             diff_acp(4,2,im) = diff_acp(4,2,im) + p*matelem(+1, 0,-1)
             diff_acp(4,1,im) = diff_acp(4,1,im) + p*matelem( 0, 0,-1)
             diff_acp(5,2,im) = diff_acp(5,2,im) + p*matelem(+1,-1,-1)
             diff_acp(5,1,im) = diff_acp(5,1,im) + p*matelem( 0,-1,-1)
             diff_acp(6,2,im) = diff_acp(6,2,im) + p*matelem(+1,+1,-1)
             diff_acp(6,1,im) = diff_acp(6,1,im) + p*matelem( 0,+1,-1)
             diff_acp(7,2,im) = diff_acp(7,2,im) + p*matelem(+1, 0,+1)
             diff_acp(7,1,im) = diff_acp(7,1,im) + p*matelem( 0, 0,+1)
             diff_acp(8,2,im) = diff_acp(8,2,im) + p*matelem(+1,-1,+1)
             diff_acp(8,1,im) = diff_acp(8,1,im) + p*matelem( 0,-1,+1)
             diff_acp(9,2,im) = diff_acp(9,2,im) + p*matelem(+1,+1,+1)
             diff_acp(9,1,im) = diff_acp(9,1,im) + p*matelem( 0,+1,+1)
          endif
          diff_acp(1,2,im)= diff_acp(1,2,im) + diff_qxr(iy,iz)/eta
          diff_rhs(im)    = diff_rhs(im)     + diff_rxr(iy,iz)
       enddo
    enddo
    !
    ! ... Y boundaries
    !
    if(diff_dimy) then
    do iz=izb,ize
       do ix=ixb,ixe
          !
          ! Left BC
          !              
          iy = 1
          !
          ! First get the eta-function
          !
          x     = diff_x(ix)
          y     = diff_y(iy)
          z     = diff_z(iz)
          call eta_function(diff_imetric,x,y,z,eta)
          !
          ! Now install the BC
          !
          call find_components_hflux(2,-1,ix,iy+1,iz,matelem)
          im = (ix-1) + (iy-1)*diff_nx + (iz-1)*diff_nx*diff_ny + 1
          p  = diff_pyl(ix,iz)
          diff_acp(1,2,im) = diff_acp(1,2,im) + p*matelem( 0,-1, 0)
          diff_acp(3,2,im) = diff_acp(3,2,im) + p*matelem( 0, 0, 0)
          diff_acp(1,1,im) = diff_acp(1,1,im) + p*matelem(-1,-1, 0)
          diff_acp(3,1,im) = diff_acp(3,1,im) + p*matelem(-1, 0, 0)
          diff_acp(1,3,im) = diff_acp(1,3,im) + p*matelem(+1,-1, 0)
          diff_acp(3,3,im) = diff_acp(3,3,im) + p*matelem(+1, 0, 0)
          if(diff_dimz) then 
             diff_acp(4,2,im) = diff_acp(4,2,im) + p*matelem( 0,-1,-1)
             diff_acp(6,2,im) = diff_acp(6,2,im) + p*matelem( 0, 0,-1)
             diff_acp(4,1,im) = diff_acp(4,1,im) + p*matelem(-1,-1,-1)
             diff_acp(6,1,im) = diff_acp(6,1,im) + p*matelem(-1, 0,-1)
             diff_acp(4,3,im) = diff_acp(4,3,im) + p*matelem(+1,-1,-1)
             diff_acp(6,3,im) = diff_acp(6,3,im) + p*matelem(+1, 0,-1)
             diff_acp(7,2,im) = diff_acp(7,2,im) + p*matelem( 0,-1,+1)
             diff_acp(9,2,im) = diff_acp(9,2,im) + p*matelem( 0, 0,+1)
             diff_acp(7,1,im) = diff_acp(7,1,im) + p*matelem(-1,-1,+1)
             diff_acp(9,1,im) = diff_acp(9,1,im) + p*matelem(-1, 0,+1)
             diff_acp(7,3,im) = diff_acp(7,3,im) + p*matelem(+1,-1,+1)
             diff_acp(9,3,im) = diff_acp(9,3,im) + p*matelem(+1, 0,+1)
          endif
          diff_acp(1,2,im)= diff_acp(1,2,im) + diff_qyl(ix,iz)/eta
          diff_rhs(im)    = diff_rhs(im)     + diff_ryl(ix,iz)
          !
          ! Right BC
          !
          iy = diff_ny
          !
          ! First get the eta-function
          !
          x     = diff_x(ix)
          y     = diff_y(iy)
          z     = diff_z(iz)
          call eta_function(diff_imetric,x,y,z,eta)
          !
          ! Now install the BC
          !
          call find_components_hflux(2,+1,ix,iy-1,iz,matelem)
          im = (ix-1) + (iy-1)*diff_nx + (iz-1)*diff_nx*diff_ny + 1
          p  = diff_pyr(ix,iz)
          diff_acp(1,2,im) = diff_acp(1,2,im) + p*matelem( 0,+1, 0)
          diff_acp(2,2,im) = diff_acp(2,2,im) + p*matelem( 0, 0, 0)
          diff_acp(1,1,im) = diff_acp(1,1,im) + p*matelem(-1,+1, 0)
          diff_acp(2,1,im) = diff_acp(2,1,im) + p*matelem(-1, 0, 0)
          diff_acp(1,3,im) = diff_acp(1,3,im) + p*matelem(+1,+1, 0)
          diff_acp(2,3,im) = diff_acp(2,3,im) + p*matelem(+1, 0, 0)
          if(diff_dimz) then 
             diff_acp(4,2,im) = diff_acp(4,2,im) + p*matelem( 0,+1,-1)
             diff_acp(5,2,im) = diff_acp(5,2,im) + p*matelem( 0, 0,-1)
             diff_acp(4,1,im) = diff_acp(4,1,im) + p*matelem(-1,+1,-1)
             diff_acp(5,1,im) = diff_acp(5,1,im) + p*matelem(-1, 0,-1)
             diff_acp(4,3,im) = diff_acp(4,3,im) + p*matelem(+1,+1,-1)
             diff_acp(5,3,im) = diff_acp(5,3,im) + p*matelem(+1, 0,-1)
             diff_acp(7,2,im) = diff_acp(7,2,im) + p*matelem( 0,+1,+1)
             diff_acp(8,2,im) = diff_acp(8,2,im) + p*matelem( 0, 0,+1)
             diff_acp(7,1,im) = diff_acp(7,1,im) + p*matelem(-1,+1,+1)
             diff_acp(8,1,im) = diff_acp(8,1,im) + p*matelem(-1, 0,+1)
             diff_acp(7,3,im) = diff_acp(7,3,im) + p*matelem(+1,+1,+1)
             diff_acp(8,3,im) = diff_acp(8,3,im) + p*matelem(+1, 0,+1)
          endif
          diff_acp(1,2,im)= diff_acp(1,2,im) + diff_qyr(ix,iz)/eta
          diff_rhs(im)    = diff_rhs(im)     + diff_ryr(ix,iz)
       enddo
    enddo
    endif
    !
    ! ... Z boundaries
    ! 
    !        >>>> NOT YET COMPLETELY READY <<<<
    !
    if(diff_dimz) then
    write(*,*) 'WARNING: Z-BOUNDARY CONDITION STILL PRIMITIVE!!!'
    write(*,*) 'For serious use: complete the code at this point!'
    do iy=iyb,iye
       do ix=ixb,ixe
          !
          ! Left BC
          !
          iz = 1
          !
          ! First get the eta-function
          !
          x     = diff_x(ix)
          y     = diff_y(iy)
          z     = diff_z(iz)
          call eta_function(diff_imetric,x,y,z,eta)
          !
          ! Now install the BC
          !
          call find_components_hflux(3,-1,ix,iy,iz+1,matelem)
          im = (ix-1) + (iy-1)*diff_nx + (iz-1)*diff_nx*diff_ny + 1
          p  = diff_pzl(ix,iy)
          diff_acp(1,2,im)= diff_acp(1,2,im) + p*matelem(0,0,-1)
          diff_acp(8,2,im)= diff_acp(8,2,im) + p*matelem(0,0, 0)
          diff_acp(1,2,im)= diff_acp(1,2,im) + diff_qzl(ix,iy)/eta
          diff_rhs(im)    = diff_rhs(im)     + diff_rzl(ix,iy)
          !
          ! Right BC
          !
          iz = diff_nz
          !
          ! First get the eta-function
          !
          x     = diff_x(ix)
          y     = diff_y(iy)
          z     = diff_z(iz)
          call eta_function(diff_imetric,x,y,z,eta)
          !
          ! Now install the BC
          !
          call find_components_hflux(3,+1,ix,iy,iz-1,matelem)
          im = (ix-1) + (iy-1)*diff_nx + (iz-1)*diff_nx*diff_ny + 1
          p  = diff_pzr(ix,iy)
          diff_acp(1,2,im)= diff_acp(1,2,im) + p*matelem(0,0,+1)
          diff_acp(5,2,im)= diff_acp(5,2,im) + p*matelem(0,0, 0)
          diff_acp(1,2,im)= diff_acp(1,2,im) + diff_qzr(ix,iy)/eta
          diff_rhs(im)    = diff_rhs(im)     + diff_rzr(ix,iy)
       enddo
    enddo
    endif
    !
    ! Now the ribbons: average
    !
    if(diff_dimy) then
       do iz=izb,ize
          ix = 1
          iy = 1
          im = (ix-1) + (iy-1)*diff_nx + (iz-1)*diff_nx*diff_ny + 1
          diff_acp(1,2,im) = 2.d0
          diff_acp(1,3,im) = -1.0d0
          diff_acp(3,2,im) = -1.0d0
          ix = 1
          iy = diff_ny
          im = (ix-1) + (iy-1)*diff_nx + (iz-1)*diff_nx*diff_ny + 1
          diff_acp(1,2,im) = 2.d0
          diff_acp(1,3,im) = -1.d0
          diff_acp(2,2,im) = -1.d0
          ix = diff_nx
          iy = 1
          im = (ix-1) + (iy-1)*diff_nx + (iz-1)*diff_nx*diff_ny + 1
          diff_acp(1,2,im) = 2.d0
          diff_acp(1,1,im) = -1.d0
          diff_acp(3,2,im) = -1.d0
          ix = diff_nx
          iy = diff_ny
          im = (ix-1) + (iy-1)*diff_nx + (iz-1)*diff_nx*diff_ny + 1
          diff_acp(1,2,im) = 2.d0
          diff_acp(1,1,im) = -1.d0
          diff_acp(2,2,im) = -1.d0
       enddo
    endif
    if(diff_dimz) then
       do ix=ixb,ixe
          iy = 1
          iz = 1
          im = (ix-1) + (iy-1)*diff_nx + (iz-1)*diff_nx*diff_ny + 1
          diff_acp(1,2,im) = 2.d0
          diff_acp(3,2,im) = -1.d0
          diff_acp(7,2,im) = -1.d0
          iy = diff_ny
          iz = 1
          im = (ix-1) + (iy-1)*diff_nx + (iz-1)*diff_nx*diff_ny + 1
          diff_acp(1,2,im) = 2.d0
          diff_acp(2,2,im) = -1.d0
          diff_acp(7,2,im) = -1.d0
          iy = 1
          iz = diff_nz
          im = (ix-1) + (iy-1)*diff_nx + (iz-1)*diff_nx*diff_ny + 1
          diff_acp(1,2,im) = 2.d0
          diff_acp(3,2,im) = -1.d0
          diff_acp(4,2,im) = -1.d0
          iy = diff_ny
          iz = diff_nz
          im = (ix-1) + (iy-1)*diff_nx + (iz-1)*diff_nx*diff_ny + 1
          diff_acp(1,2,im) = 2.d0
          diff_acp(2,2,im) = -1.d0
          diff_acp(4,2,im) = -1.d0
       enddo
       do iy=iyb,iye
          ix = 1
          iz = 1
          im = (ix-1) + (iy-1)*diff_nx + (iz-1)*diff_nx*diff_ny + 1
          diff_acp(1,2,im) = 2.d0
          diff_acp(1,3,im) = -1.d0
          diff_acp(7,2,im) = -1.d0              
          ix = diff_nx
          iz = 1
          im = (ix-1) + (iy-1)*diff_nx + (iz-1)*diff_nx*diff_ny + 1
          diff_acp(1,2,im) = 2.d0
          diff_acp(1,1,im) = -1.d0
          diff_acp(7,2,im) = -1.d0              
          ix = 1
          iz = diff_nz
          im = (ix-1) + (iy-1)*diff_nx + (iz-1)*diff_nx*diff_ny + 1
          diff_acp(1,2,im) = 2.d0
          diff_acp(1,3,im) = -1.d0
          diff_acp(4,2,im) = -1.d0              
          ix = diff_nx
          iz = diff_nz
          im = (ix-1) + (iy-1)*diff_nx + (iz-1)*diff_nx*diff_ny + 1
          diff_acp(1,2,im) = 2.d0
          diff_acp(1,1,im) = -1.d0
          diff_acp(4,2,im) = -1.d0              
       enddo
    endif
    !
    ! Now the corners: average
    !
    if(diff_dimz) then
       ix = 1
       iy = 1
       iz = 1
       diff_acp(1,2,im) =  3.d0
       diff_acp(1,3,im) = -1.d0
       diff_acp(3,2,im) = -1.d0
       diff_acp(7,2,im) = -1.d0
       ix = diff_nx
       iy = 1
       iz = 1
       diff_acp(1,2,im) =  3.d0
       diff_acp(1,1,im) = -1.d0
       diff_acp(3,2,im) = -1.d0
       diff_acp(7,2,im) = -1.d0
       ix = 1
       iy = diff_ny
       iz = 1
       diff_acp(1,2,im) =  3.d0
       diff_acp(1,3,im) = -1.d0
       diff_acp(2,2,im) = -1.d0
       diff_acp(7,2,im) = -1.d0
       ix = 1
       iy = 1
       iz = diff_nz
       diff_acp(1,2,im) =  3.d0
       diff_acp(1,3,im) = -1.d0
       diff_acp(3,2,im) = -1.d0
       diff_acp(4,2,im) = -1.d0
       ix = diff_nx
       iy = diff_ny
       iz = 1
       diff_acp(1,2,im) =  3.d0
       diff_acp(1,1,im) = -1.d0
       diff_acp(2,2,im) = -1.d0
       diff_acp(7,2,im) = -1.d0
       ix = diff_nx
       iy = 1
       iz = diff_nz
       diff_acp(1,2,im) =  3.d0
       diff_acp(1,1,im) = -1.d0
       diff_acp(3,2,im) = -1.d0
       diff_acp(4,2,im) = -1.d0
       ix = diff_nx
       iy = diff_ny
       iz = diff_nz
       diff_acp(1,2,im) =  3.d0
       diff_acp(1,1,im) = -1.d0
       diff_acp(2,2,im) = -1.d0
       diff_acp(4,2,im) = -1.d0
    endif
    !
    ! Now normalize the rows of matrix and vector, which is in fact
    ! a very simple kind of preconditioning
    !
    call normalize_rows(diff_lda,diff_nband,3)
    !
    ! Done...
    !
  end subroutine create_the_band_matrix_and_diff_rhs


  !---------------------------------------------------------------
  !           COMPUTE MATRIX ELEMENTS FOR DIFFUSION EQ
  !
  ! It will compute the matrix elements corresponding to the
  ! following equation:
  !
  !   (H^x*S^x)_ix+1/2,iy,iz - (H^x*S^x)_ix-1/2,iy,iz +
  !   (H^y*S^y)_ix,iy+1/2,iz - (H^y*S^y)_ix,iy-1/2,iz +
  !   (H^z*S^z)_iz,iy,iz+1/2 - (H^z*S^z)_ix,iy,iz-1/2
  !     = V_ix,iy,iz * ( s - beta*J )
  !
  !---------------------------------------------------------------
  subroutine find_matrix_elements_pde(ix,iy,iz,matelem)
    implicit none
    integer :: ix,iy,iz
    doubleprecision :: matelem(-1:1,-1:1,-1:1)
    doubleprecision :: comp_p(-1:1,-1:1,-1:1)
    doubleprecision :: comp_m(-1:1,-1:1,-1:1)
    doubleprecision :: vol,surf_p,surf_m
    doubleprecision :: x,y,z,eta
    integer :: k,l,m
    !
    ! First checks on the indices
    !
    if((ix.lt.2).or.(ix.gt.diff_nx-1)) stop 61
    if(diff_idim.eq.3) then
        if((iz.lt.2).or.(iz.gt.diff_nz-1)) stop 63
    else
        if(iz.ne.1) stop 73
    endif
    if(diff_idim.ge.2) then
        if((iy.lt.2).or.(iy.gt.diff_ny-1)) stop 62
    else
        if(iy.ne.1) stop 72
    endif
    !
    ! First put everything to zero
    !
    do k=-1,1
       do l=-1,1
          do m=-1,1
             matelem(k,l,m) = 0.d0
          enddo
       enddo
    enddo
    !
    ! Get the volume element
    !      
    vol = get_volume_element(ix,iy,iz)
    !     
    ! Flux differences in X-direction
    !
    call find_components_hflux(1,1,ix,iy,iz,comp_p)
    call find_components_hflux(1,-1,ix,iy,iz,comp_m)
    surf_p = get_surface_element(1,2*ix+1,2*iy,2*iz)
    surf_m = get_surface_element(1,2*ix-1,2*iy,2*iz)
    do k=-1,1
       do l=-1,1
          do m=-1,1
             matelem(k,l,m) = matelem(k,l,m) + comp_p(k,l,m) * surf_p * diff_sgnx
             matelem(k,l,m) = matelem(k,l,m) - comp_m(k,l,m) * surf_m * diff_sgnx
          enddo
       enddo
    enddo
    !
    ! Flux differences in Y-direction
    !
    if(diff_dimy) then
    call find_components_hflux(2,1,ix,iy,iz,comp_p)
    call find_components_hflux(2,-1,ix,iy,iz,comp_m)
    surf_p = get_surface_element(2,2*ix,2*iy+1,2*iz)
    surf_m = get_surface_element(2,2*ix,2*iy-1,2*iz)
    do k=-1,1
       do l=-1,1
          do m=-1,1
             matelem(k,l,m) = matelem(k,l,m) + comp_p(k,l,m) * surf_p * diff_sgny
             matelem(k,l,m) = matelem(k,l,m) - comp_m(k,l,m) * surf_m * diff_sgny
          enddo
       enddo
    enddo
    endif
    !
    ! Flux differences in Z-direction
    !
    if(diff_dimz) then
    call find_components_hflux(3,1,ix,iy,iz,comp_p)
    call find_components_hflux(3,-1,ix,iy,iz,comp_m)
    surf_p = get_surface_element(3,2*ix,2*iy,2*iz+1)
    surf_m = get_surface_element(3,2*ix,2*iy,2*iz-1)
    do k=-1,1
       do l=-1,1
          do m=-1,1
             matelem(k,l,m) = matelem(k,l,m) + comp_p(k,l,m) * surf_p * diff_sgnz
             matelem(k,l,m) = matelem(k,l,m) - comp_m(k,l,m) * surf_m * diff_sgnz
          enddo
       enddo
    enddo
    endif
    !
    ! Get the eta-function
    !
    x     = diff_x(ix)
    y     = diff_y(iy)
    z     = diff_z(iz)
    call eta_function(diff_imetric,x,y,z,eta)
    !
    ! Now the source term part that is proportional to J
    !     
    matelem(0,0,0) = matelem(0,0,0) + diff_b(ix,iy,iz)*vol/eta
    !
    ! Now divide by volume of the cell
    !
    do k=-1,1
       do l=-1,1
          do m=-1,1
             matelem(k,l,m) = matelem(k,l,m) / vol
          enddo
       enddo
    enddo
  end subroutine find_matrix_elements_pde



  !---------------------------------------------------------------
  !                FIND THE COMPONENTS FOR THE FLUX
  !---------------------------------------------------------------
  subroutine find_components_hflux(idir,iplus,ix,iy,iz,comp)
    implicit none
    integer :: idir,iplus,ix,iy,iz
    doubleprecision :: comp(-1:1,-1:1,-1:1)
    doubleprecision :: dx,dy,dz
    doubleprecision :: alpha,x,y,z,sqgii
    doubleprecision :: eta
    integer :: ix2,iy2,iz2,ip,im,k,l,m
    !
    ! Some useful stuff
    !
    if(iplus.eq.1) then
        ip = 1
        im = 0
    elseif(iplus.eq.-1) then
        ip = 0
        im = -1
    else
        stop 76
    endif
    !
    ! Get the half-integers
    !
    if(idir.eq.1) then
        ix2 = 2*ix + iplus
        iy2 = 2*iy
        iz2 = 2*iz
    elseif(idir.eq.2) then
        ix2 = 2*ix
        iy2 = 2*iy + iplus
        iz2 = 2*iz
    elseif(idir.eq.3) then
        ix2 = 2*ix
        iy2 = 2*iy
        iz2 = 2*iz + iplus
    else
        stop 57
    endif
    !
    ! Get the alpha
    !
    alpha = get_alpha(idir,ix2,iy2,iz2)
    !
    ! Get the dx, dy and dz
    !
    dx = get_dgrid(1,ix2)
    if(diff_dimy) then
        dy = get_dgrid(2,iy2) 
    else 
        dy = 0.d0
    endif
    if(diff_dimz) then
        dz = get_dgrid(3,iz2) 
    else
        dz = 0.d0
    endif
    !
    ! First put everything to zero
    !
    do k=-1,1
        do l=-1,1
            do m=-1,1
                comp(k,l,m) = 0.d0
            enddo
        enddo
    enddo
    !
    ! Do the main stuff
    !
    if(idir.eq.1) then
       !
       ! Get the sqrt(g_11)
       !
       x     = 0.5d0 * ( diff_x(ix+ip) + diff_x(ix+im) )
       y     = diff_y(iy)
       z     = diff_z(iz)
       sqgii = sqroot_gij(diff_imetric,idir,x,y,z)
       !
       ! Add the main derivative
       !
       comp(ip,0,0)  = comp(ip,0,0)  - get_fij(1,1,ix+ip,iy,iz) / dx
       comp(im,0,0)  = comp(im,0,0)  + get_fij(1,1,ix+im,iy,iz) / dx
       !
       ! Add the perpendicular derivative Y
       !
       if(diff_dimy) then
       comp(ip,+1,0) = comp(ip,+1,0) - 0.5d0 * get_fij(1,2,ix+ip,iy+1,iz) / dy
       comp(im,+1,0) = comp(im,+1,0) - 0.5d0 * get_fij(1,2,ix+im,iy+1,iz) / dy
       comp(ip,-1,0) = comp(ip,-1,0) + 0.5d0 * get_fij(1,2,ix+ip,iy-1,iz) / dy
       comp(im,-1,0) = comp(im,-1,0) + 0.5d0 * get_fij(1,2,ix+im,iy-1,iz) / dy
       endif
       !
       ! Add the perpendicular derivative Z
       !
       if(diff_dimz) then
       comp(ip,0,+1) = comp(ip,0,+1) - 0.5d0 * get_fij(1,3,ix+ip,iy,iz+1) / dz
       comp(im,0,+1) = comp(im,0,+1) - 0.5d0 * get_fij(1,3,ix+im,iy,iz+1) / dz
       comp(ip,0,-1) = comp(ip,0,-1) + 0.5d0 * get_fij(1,3,ix+ip,iy,iz-1) / dz
       comp(im,0,-1) = comp(im,0,-1) + 0.5d0 * get_fij(1,3,ix+im,iy,iz-1) / dz
       endif
       !
       ! Add the covariant derivative terms
       !
       comp(ip,0,0)  = comp(ip,0,0)  - 0.5d0 * get_connection(idir,ix+ip,iy,iz)
       comp(im,0,0)  = comp(im,0,0)  - 0.5d0 * get_connection(idir,ix+im,iy,iz)
       !
    elseif(idir.eq.2) then
       !
       ! Get the sqrt(g_22)
       !
       x     = diff_x(ix)
       y     = 0.5d0 * ( diff_y(iy+ip) + diff_y(iy+im) )
       z     = diff_z(iz)
       sqgii = sqroot_gij(diff_imetric,idir,x,y,z)
       !
       ! Add the main derivative
       !
       comp(0,ip,0)  = comp(0,ip,0)  - get_fij(2,2,ix,iy+ip,iz) / dy
       comp(0,im,0)  = comp(0,im,0)  + get_fij(2,2,ix,iy+im,iz) / dy
       !
       ! Add the perpendicular derivative X
       !
       comp(+1,ip,0) = comp(+1,ip,0) - 0.5d0 * get_fij(2,1,ix+1,iy+ip,iz) / dx
       comp(+1,im,0) = comp(+1,im,0) - 0.5d0 * get_fij(2,1,ix+1,iy+im,iz) / dx
       comp(-1,ip,0) = comp(-1,ip,0) + 0.5d0 * get_fij(2,1,ix-1,iy+ip,iz) / dx
       comp(-1,im,0) = comp(-1,im,0) + 0.5d0 * get_fij(2,1,ix-1,iy+im,iz) / dx
       !
       ! Add the perpendicular derivative Z
       !          
       if(diff_dimz) then
       comp(0,ip,+1) = comp(0,ip,+1) - 0.5d0 * get_fij(2,3,ix,iy+ip,iz+1) / dz
       comp(0,im,+1) = comp(0,im,+1) - 0.5d0 * get_fij(2,3,ix,iy+im,iz+1) / dz
       comp(0,ip,-1) = comp(0,ip,-1) + 0.5d0 * get_fij(2,3,ix,iy+ip,iz-1) / dz
       comp(0,im,-1) = comp(0,im,-1) + 0.5d0 * get_fij(2,3,ix,iy+im,iz-1) / dz
       endif
       !
       ! Add the covariant derivative terms
       !
       comp(0,ip,0)  = comp(0,ip,0)  - 0.5d0 * get_connection(idir,ix,iy+ip,iz)
       comp(0,im,0)  = comp(0,im,0)  - 0.5d0 * get_connection(idir,ix,iy+im,iz)
       !
    elseif(idir.eq.3) then
       !
       ! Get the sqrt(g_33)
       !
       x     = diff_x(ix)
       y     = diff_y(iy)
       z     = 0.5d0 * ( diff_z(iz+ip) + diff_z(iz+im) )
       sqgii = sqroot_gij(diff_imetric,idir,x,y,z)
       !
       ! Add the main derivative
       !
       comp(0,0,ip)  = comp(0,0,ip)  - get_fij(3,3,ix,iy,iz+ip) / dz
       comp(0,0,im)  = comp(0,0,im)  + get_fij(3,3,ix,iy,iz+im) / dz
       !
       ! Add the perpendicular derivative X
       !
       comp(+1,0,ip) = comp(+1,0,ip) - 0.5d0 * get_fij(3,1,ix+1,iy,iz+ip) / dx
       comp(+1,0,im) = comp(+1,0,im) - 0.5d0 * get_fij(3,1,ix+1,iy,iz+im) / dx
       comp(-1,0,ip) = comp(-1,0,ip) + 0.5d0 * get_fij(3,1,ix-1,iy,iz+ip) / dx
       comp(-1,0,im) = comp(-1,0,im) + 0.5d0 * get_fij(3,1,ix-1,iy,iz+im) / dx
       !
       ! Add the perpendicular derivative Y
       !         
       comp(0,+1,ip) = comp(0,+1,ip) - 0.5d0 * get_fij(3,2,ix,iy+1,iz+ip) / dy
       comp(0,+1,im) = comp(0,+1,im) - 0.5d0 * get_fij(3,2,ix,iy+1,iz+im) / dy
       comp(0,-1,ip) = comp(0,-1,ip) + 0.5d0 * get_fij(3,2,ix,iy-1,iz+ip) / dy
       comp(0,-1,im) = comp(0,-1,im) + 0.5d0 * get_fij(3,2,ix,iy-1,iz+im) / dy
       !
       ! Add the covariant derivative terms
       !
       comp(0,0,ip)  = comp(0,0,ip)  - 0.5d0 * get_connection(idir,ix,iy,iz+ip)
       comp(0,0,im)  = comp(0,0,im)  - 0.5d0 * get_connection(idir,ix,iy,iz+im)
       !
    else
       stop 57
    endif
    !
    ! Multiply everything with sqrt(g_ii)/alpha
    !
    do k=-1,1
       do l=-1,1
          do m=-1,1
             comp(k,l,m) = (sqgii/alpha) * comp(k,l,m)
          enddo
       enddo
    enddo
    !
  end subroutine find_components_hflux



  !===============================================================
  !            ROUTINES FOR THE TIME-DEPENDENT PART
  !
  ! Note that the time-dependent integration is implicit, but
  ! only linearly implicit. The change of the g-factor, which
  ! encodes the relation between mean intensity J and temperature T
  ! is not included in the implicit differencing: it is computed
  ! from the last time step.
  !
  ! We use CGS units for the natural constants.
  !===============================================================


  !---------------------------------------------------------------
  !         MATRIX-ELEMENTS AND RHS FOR TIME-DEPENDENCE
  !
  ! We follow here Kuiper, Klahr, Dullemond, Kley & Henning (2010)
  ! but instead of E_R we use the mean intensity J = c*E_R/(4*pi).
  ! And instead of f_c we use g = 1 / ( c * f_c ). The extra factor
  ! 1/c comes because (1/f_c) dE_r/dt = (1/f_c) * (4*pi/c) * dJ/dt
  ! The 4*pi factor is divided out (we use H instead of F) and the
  ! factor 1/c is absorbed in g. 
  !---------------------------------------------------------------
  subroutine add_time_dependence_terms(timestep)
    implicit none
    double precision :: timestep,dtinverse
    double precision, parameter :: cc = 2.9979245800000d10        ! = light speed
    double precision, parameter :: aa = 7.5657d-15                ! = 4 * sigma_SB / cc
    double precision :: gdtinv
    integer :: ix,iy,iz
    integer :: ixb,ixe,iyb,iye,izb,ize,im
    !
    ! First find out the boundaries within which to fill the matrix with
    ! the PDE. If a dimension is active, then we must skip the boundary
    ! cells. If not, then use index 1 simply. 
    !
    ixb = 2
    ixe = diff_nx - 1
    if(diff_dimy) then
       iyb = 2
       iye = diff_ny - 1
    else
       iyb = 1
       iye = 1
    endif
    if(diff_dimz) then
       izb = 2
       ize = diff_nz - 1
    else
       izb = 1
       ize = 1
    endif
    !
    ! Compute the inverse time step
    !
    dtinverse = 1.d0 / timestep
    !
    ! Compute the temperature everywhere from the current J
    ! Kuiper et al. Eq. 16. The diff_t_irrad**4 is the second
    ! term on the rhs of Eq. 16 (div by a), where we defined the 
    ! irrad temperature the solution to Eq. 16 when E_R=0.
    ! The caller of this package has to calculate diff_t_irrad
    ! him/herself from e.g. the stellar F_*. Note that the
    ! source term in the PDE we solve here (see top of this
    ! file) should be the divergence of this F_*, i.e. the
    ! diff_s(:,:,;) must be consistent with diff_r_irrad(:,:,:).
    !
    diff_tgas(:,:,:) = ( diff_j(:,:,:)*pisigsb +                         &
                         ( diff_t_irrad(:,:,:) + diff_tmin )**4 )**0.25
    !
    ! Calculate the g factor at the current time
    !
    !   g = 1 / ( c * f_c )  with f_c defined below Eq. 6 of Kuiper etal
    !
    diff_gfact(:,:,:) = ( 1.d0 + diff_cvrho(:,:,:) /                     &
                          ( 4 * aa * diff_tgas(:,:,:)**3 )  ) / cc
    !
    ! Now a loop over the domain, apart from the boundaries, to add
    ! the terms for time-dependence to the matrix and rhs
    !
    do iz=izb,ize
       do iy=iyb,iye
          do ix=ixb,ixe
             im = (ix-1) + (iy-1)*diff_nx + (iz-1)*diff_nx*diff_ny + 1
             !
             ! Get g / dt
             !
             gdtinv = diff_gfact(ix,iy,iz) * dtinverse
             !
             ! Add g/dt to matrix elements
             !
             diff_acp(1,2,im) = diff_acp(1,2,im) + gdtinv
             !
             ! Add (g/dt)*J to rhs
             !
             diff_rhs(im) = diff_rhs(im) + gdtinv * diff_j(ix,iy,iz)
          enddo
       enddo
    enddo
    !
  end subroutine add_time_dependence_terms
  


  
  !===============================================================
  !      ROUTINES FOR FINDING PROPER AVERAGED QUANTITIES 
  !
  ! The following routines find the value of several geometric
  ! quantities at cell centers, at cell boundaries, or at the
  ! cell ribbons. For gridded quantities (such as the Eddington
  ! factors) suitable averages are used; for fundamental geometric
  ! quantities (such as metric, sufrace elements, etc), analytic
  ! formulae are used, in order to get the geometric stuff as
  ! accurate as possible. 
  !===============================================================


  !---------------------------------------------------------------
  !      FIND THE F^{IJ} EDDINGTON FACTOR AT A GIVEN POSITION
  !
  ! This routine returns the value of the Eddington factor at 
  ! cell center.
  !
  ! ARGUMENTS:
  !     i,j         = The i,j of f^{i,j}
  !     ix,iy,iz    = The grid cell center position
  !
  ! NOTE: 
  !   If the f^{ij} are being asked at the boundary cells, then
  !   zero is returned, because there is no proper way of imposing
  !   boundary conditions with f^{12}.eq.0. 
  !
  ! NOTE: 
  !   It returns the covariant form of f^ij, eventhough the local
  !   form of the f^ij are stored in the arrays.
  !
  ! NOTE:
  !   In fact, this function returns eta*f^ij instead of only
  !   f^ij. The eta-function is a well-chosen renormalization
  !   function such that the solution eta*J is a function that
  !   does not change very much. For instance, for spherical
  !   coordinates this is R^2. See the subroutine eta_function().
  !
  !---------------------------------------------------------------
  function get_fij(ii,jj,ix,iy,iz)
    implicit none
    integer :: idum,i,j,ix,iy,iz,ii,jj
    doubleprecision :: x,y,z,eta,get_fij
    !
    i = ii
    j = jj
    !
    ! x,y,z
    !
    x   = diff_x(ix)
    y   = diff_y(iy)
    z   = diff_z(iz)
    !
    ! Ascending order
    !
    if(j.lt.i) then
       idum=j
       j=i
       i=idum
    endif
    !
    ! If wrong components, then error
    !
    if((i.gt.3).or.(j.gt.3).or.(i.lt.1).or.(j.lt.1)) then
       write(*,*) 'Eddington factor indices out of bound'
       stop 13
    endif
    !
    ! If out of bounds, then error
    !
    if((ix.gt.diff_nx).or.(ix.lt.1)) then
       write(*,*) 'Internal error: ix out of bounds'
       write(*,*) ix
       stop 13
    endif
    if((iy.gt.diff_ny).or.(iy.lt.1)) then
       write(*,*) 'Internal error: iy out of bounds'
       write(*,*) iy
       stop 13
    endif
    if((iz.gt.diff_nz).or.(iz.lt.1)) then
       write(*,*) 'Internal error: iz out of bounds'
       write(*,*) iz
       stop 13
    endif
    !
    ! Return the usual values
    !
    if((i.eq.1).and.(j.eq.1)) then
       get_fij = diff_fxx(ix,iy,iz) 
    elseif((i.eq.1).and.(j.eq.2)) then
       get_fij = diff_fxy(ix,iy,iz) 
    elseif((i.eq.1).and.(j.eq.3)) then
       get_fij = diff_fxz(ix,iy,iz) 
    elseif((i.eq.2).and.(j.eq.2)) then
       get_fij = diff_fyy(ix,iy,iz) 
    elseif((i.eq.2).and.(j.eq.3)) then
       get_fij = diff_fyz(ix,iy,iz) 
    elseif((i.eq.3).and.(j.eq.3)) then
       get_fij = diff_fzz(ix,iy,iz)
    else
       stop 653
    endif
    !
    ! Divide by the eta-function, because in the PDE we have a differential
    ! operator acting on f^ij*J. Since the sol vector is sol=eta*J we
    ! can write f^ij*J = (f^ij/eta)*sol.
    !
    call eta_function(diff_imetric,x,y,z,eta)
    get_fij = get_fij / eta
    !
    return
  end function get_fij


  !---------------------------------------------------------------
  !              FIND THE ALPHA AT CELL SURFACE
  !
  !
  ! ARGUMENTS:
  !     idir        = Direction of alpha: alpha_x, or _y or _z?
  !     ix2,iy2,iz2 = Twice the grid position, plus 0 or 1 when
  !                      in cell center ix2=2*(ix) or cell boundary 
  !                      ix2=2*(ix+1/2). Note that for e.g. idir.eq.1
  !                      the ix2 must be odd while the iy2 must be
  !                      even.
  !
  !---------------------------------------------------------------
  function get_alpha(idir,ix2,iy2,iz2)
    implicit none
    integer :: ix2,iy2,iz2,ix,iy,iz,ixp,iyp,izp,idir
    doubleprecision :: get_alpha
    !
    ! Find out which cell boundary
    !
    ix=ix2/2
    iy=iy2/2
    iz=iz2/2
    if(.not.even(ix2)) then
       ixp=ix2/2+1
    else
       ixp=ix2/2
    endif
    if(.not.even(iy2)) then
       iyp=iy2/2+1
    else
       iyp=iy2/2
    endif
    if(.not.even(iz2)) then
       izp=iz2/2+1
    else
       izp=iz2/2
    endif
    !
    ! If out of bounds, then error
    !
    if((ix2.ge.2*diff_nx+1).or.(iy2.ge.2*diff_ny+1).or.   &
       (iz2.ge.2*diff_nz+1).or.(ix2.le.1).or.(iy2.le.1).or.(iz2.le.1)) then
       write(*,*) 'Internal error: ix2,iy2,iz2 out of bounds'
       write(*,*) 'ix2=',ix2,' iy2=',iy2,' iz2=',iz2
       write(*,*) 2*diff_nx+1,2*diff_ny+1,2*diff_nz+1
       stop 15
    endif
    !
    ! Return the alpha
    !
    if(idir.eq.1) then
       get_alpha = 0.5d0 * ( diff_ax(ix,iy,iz) + diff_ax(ixp,iyp,izp) )
    elseif(idir.eq.2) then
       get_alpha = 0.5d0 * ( diff_ay(ix,iy,iz) + diff_ay(ixp,iyp,izp) )
    elseif(idir.eq.3) then
       get_alpha = 0.5d0 * ( diff_az(ix,iy,iz) + diff_az(ixp,iyp,izp) )
    else
       stop 923
    endif
    !
    return
  end function get_alpha


  !---------------------------------------------------------------
  !           FIND THE GRID SPACING AT A GIVEN POSITION
  !
  ! This routine returns the grid spacing in directions 1, 2 or 3.
  ! Since the boundary conditions are imposed on the flux BETWEEN
  ! the two first and the two last grid cells in each direction,
  ! the spacing need not be requested at the real outer cell 
  ! boundaries.
  !
  ! ARGUMENTS:
  !     idir        = The direction (1,2 or 3)
  !     i2          = Twice the grid position, plus 0 or 1 when
  !                      in cell center i2=2*(i) or cell boundary 
  !                      i2=2*(i+1/2).
  !
  !---------------------------------------------------------------
  function get_dgrid(idir,i2)
    implicit none
    integer :: idir,i2,ip,im
    logical :: ev,od
    doubleprecision :: get_dgrid
    !
    ! Find out whether we are at cell boundaries or cell centers
    !
    ev=even(i2)
    od=.not.ev
    if(od) then
       ip=i2/2+1
       im=i2/2
    else
       ip=i2/2+1
       im=i2/2-1
    endif
    !
    ! If out of bounds, then error
    !
    if(idir.eq.1) then
       if((i2.ge.2*diff_nx+1).or.(i2.le.1)) then
          write(*,*) 'Internal error: i2 out of bounds'
          stop 14
       endif
    elseif(idir.eq.2) then
       if((i2.ge.2*diff_ny+1).or.(i2.le.1)) then
          write(*,*) 'Internal error: i2 out of bounds'
          stop 14
       endif
    elseif(idir.eq.3) then
       if((i2.ge.2*diff_nz+1).or.(i2.le.1)) then
          write(*,*) 'Internal error: i2 out of bounds'
          stop 14
       endif
    endif
    !
    ! If at boundary cells: dx is half as big, because there is no i+1
    ! or i-1 element.
    !
    if(idir.eq.1) then
       if(i2.eq.2) then
          get_dgrid = ( diff_x(2) - diff_x(1) )
       endif
       if(i2.eq.2*diff_nx) then
          get_dgrid = ( diff_x(diff_nx) - diff_x(diff_nx-1) )
       endif
    elseif(idir.eq.2) then
       if(diff_dimy) then
          if(i2.eq.2) then
             get_dgrid = ( diff_y(2) - diff_y(1) )
          endif
          if(i2.eq.2*diff_ny) then
             get_dgrid = ( diff_y(diff_ny) - diff_y(diff_ny-1) )
          endif
       else
          stop 771
       endif
    elseif(idir.eq.3) then
       if(diff_dimz) then
          if(i2.eq.2) then
             get_dgrid = ( diff_z(2) - diff_z(1) )
          endif
          if(i2.eq.2*diff_nz) then
             get_dgrid = ( diff_z(diff_nz) - diff_z(diff_nz-1) )
          endif
       else
          stop 772
       endif
    else 
       stop 443
    endif
    !
    ! Else return the usual values
    !
    if(idir.eq.1) then
       get_dgrid = diff_x(ip) - diff_x(im)
    elseif(idir.eq.2) then
       get_dgrid = diff_y(ip) - diff_y(im)
    elseif(idir.eq.3) then
       get_dgrid = diff_z(ip) - diff_z(im)
    else
       stop 443
    endif
    !
    return
  end function get_dgrid





  !---------------------------------------------------------------
  !               FIND THE CELL SURFACE ELEMENTS 
  !
  ! This routine returns the surface elements. 
  !
  ! ARGUMENTS:
  !     idir        = Direction in which to find the surface element
  !     ix2,iy2,iz2 = Twice the grid position, plus 0 or 1 when
  !                      in cell center ix2=2*(ix) or cell boundary 
  !                      ix2=2*(ix+1/2). Note that for e.g. idir.eq.1
  !                      the ix2 must be odd while the iy2 must be
  !                      even.
  !
  !---------------------------------------------------------------
  function get_surface_element(idir,ix2,iy2,iz2)
    implicit none
    integer :: idum,idir,ix2,iy2,iz2,ixm,iym,izm,ixp,iyp,izp
    integer :: ix,iy,iz
    doubleprecision :: x1,x2,y1,y2,z1,z2,x,y,z,surf
    doubleprecision :: get_surface_element
    logical :: evx,evy,evz,odx,ody,odz
    !
    ! Check
    !
    if(idir.gt.diff_idim) stop 99
    !
    ! Find out whether we are at cell boundaries or cell centers
    !
    evx=even(ix2)
    evy=even(iy2)
    evz=even(iz2)
    odx=.not.evx
    ody=.not.evy
    odz=.not.evz
    !
    ! If out of bounds, then error
    !
    if((ix2.gt.2*diff_nx+1).or.(iy2.gt.2*diff_ny+1).or.   &
       (iz2.gt.2*diff_nz+1).or.(ix2.lt.1).or.(iy2.lt.1).or.(iz2.lt.1)) then
       write(*,*) 'Internal error: ix2,iy2,iz2 out of bounds'
       write(*,*) 'ix2=',ix2,' iy2=',iy2,' iz2=',iz2
       stop 16
    endif
    !
    ! Find indices
    !
    ixm=ix2/2-1
    iym=iy2/2-1
    izm=iz2/2-1
    ixp=ix2/2+1
    iyp=iy2/2+1
    izp=iz2/2+1
    ix =ix2/2
    iy =iy2/2
    iz =iz2/2
    !
    ! Find x1,x2,y1,y2,z1,z2
    !
    if(idir.eq.1) then
       if(evx.or.ody.or.odz) then
          stop 6543
       endif
       ixm = ix2/2
       x1  = 0.5d0 * ( diff_x(ixm) + diff_x(ixp) )
       x2  = x1
       x   = x1
       if(diff_dimy) then
          y1 = 0.5d0 * ( diff_y(iy) + diff_y(iym) )
          y2 = 0.5d0 * ( diff_y(iy) + diff_y(iyp) )
          y  = diff_y(iy2/2)
       else
          y1 = diff_yl
          y2 = diff_yr
          y  = diff_y(1)
       endif
       if(diff_dimz) then
          z1 = 0.5d0 * ( diff_z(iz) + diff_z(izm) )
          z2 = 0.5d0 * ( diff_z(iz) + diff_z(izp) )
          z  = diff_z(iz2/2)
       else
          z1 = diff_zl
          z2 = diff_zr
          z  = diff_z(1)
       endif
    elseif(idir.eq.2) then
       if(odx.or.evy.or.odz) then
          stop 6542
       endif
       iym = iy2/2
       y1  = 0.5d0 * ( diff_y(iym) + diff_y(iyp) )
       y2  = y1
       y   = y1
       x1  = 0.5d0 * ( diff_x(ix) + diff_x(ixm) )
       x2  = 0.5d0 * ( diff_x(ix) + diff_x(ixp) )
       x   = diff_x(ix2/2)
       if(diff_dimz) then
          z1 = 0.5d0 * ( diff_z(iz) + diff_z(izm) )
          z2 = 0.5d0 * ( diff_z(iz) + diff_z(izp) )
          z  = diff_z(iz2/2)
       else
          z1 = diff_zl
          z2 = diff_zr
          z  = diff_z(1)
       endif
    elseif(idir.eq.3) then
       if(odx.or.ody.or.evz) then
          stop 6541
       endif
       izm=iz2/2
       z1  = 0.5d0 * ( diff_z(izm) + diff_z(izp) )
       z2  = z1
       z   = z1
       x1  = 0.5d0 * ( diff_x(ix) + diff_x(ixm) )
       x2  = 0.5d0 * ( diff_x(ix) + diff_x(ixp) )
       x   = diff_x(ix2/2)
       y1  = 0.5d0 * ( diff_y(iy) + diff_y(iym) )
       y2  = 0.5d0 * ( diff_y(iy) + diff_y(iyp) )
       y   = diff_y(iy2/2)
    else 
       stop 5124
    endif
    !
    ! Return the usual values
    !
    get_surface_element = surface_element(diff_imetric,idir,x1,x2,y1,y2,z1,z2)
    !
    return
  end function get_surface_element




  !---------------------------------------------------------------
  !               FIND THE CELL VOLUME ELEMENTS 
  !
  ! This routine returns the surface elements. 
  !---------------------------------------------------------------
  function get_volume_element(ix,iy,iz)
    implicit none
    integer :: ix,iy,iz
    doubleprecision :: x1,y1,z1,x2,y2,z2
    doubleprecision :: get_volume_element
    !
    ! Boundary cells must be skipped (they harbor BC's)
    !
    if((ix.eq.1).or.(ix.eq.diff_nx)) then
       write(*,*) 'Internal error: volume elements not at boundary'
       stop 13
    endif
    if(((iy.eq.1).or.(iy.eq.diff_ny)).and.diff_dimy) then
       write(*,*) 'Internal error: volume elements not at boundary'
       stop 13
    endif
    if(((iz.eq.1).or.(iz.eq.diff_nz)).and.diff_dimz) then
       write(*,*) 'Internal error: volume elements not at boundary'
       stop 13
    endif
    !
    ! Now find the coordinates of the cell boundaries
    !
    x1 = 0.5d0 * ( diff_x(ix) + diff_x(ix-1) )
    x2 = 0.5d0 * ( diff_x(ix+1) + diff_x(ix) )
    if(diff_dimy) then
       y1 = 0.5d0 * ( diff_y(iy) + diff_y(iy-1) )
       y2 = 0.5d0 * ( diff_y(iy+1) + diff_y(iy) )
    else
       y1 = diff_yl
       y2 = diff_yr
    endif
    if(diff_dimz) then
       z1 = 0.5d0 * ( diff_z(iz) + diff_z(iz-1) )
       z2 = 0.5d0 * ( diff_z(iz+1) + diff_z(iz) )
    else
       z1 = diff_zl
       z2 = diff_zr
    endif
    !
    !     Get the volume element
    !
    get_volume_element = volume_element(diff_imetric,x1,x2,y1,y2,z1,z2)
    return
  end function get_volume_element



  !---------------------------------------------------------------
  !               FIND THE CONNECTION FACTOR
  !---------------------------------------------------------------
  function get_connection(idir,ix,iy,iz)
    implicit none
    integer :: ix,iy,iz,idir
    doubleprecision :: x,y,z,fxx,fxy,fyy,fxz,fyz,fzz
    doubleprecision :: get_connection
    !
    ! Get the location
    !
    x = diff_x(ix)
    y = diff_y(iy)
    z = diff_z(iz)
    !
    ! Get the covariant Eddington tensor components
    !
    fxx = get_fij(1,1,ix,iy,iz)
    fxy = get_fij(1,2,ix,iy,iz)
    fyy = get_fij(2,2,ix,iy,iz)
    fxz = get_fij(1,3,ix,iy,iz)
    fyz = get_fij(2,3,ix,iy,iz)
    fzz = get_fij(3,3,ix,iy,iz)
    !
    ! Get the volume element
    !
    get_connection = connectionfactor(diff_imetric,idir,x,y,z,fxx,fxy,fyy,fxz,fyz,fzz)
    return
  end function get_connection




!============================================================================
!                            PART:
!                ADVECTION-DIFFUSION EQUATION
!
! This is a general purpose module for setting up and solving
! a 3-D/2-D/1-D advection-diffusion type problem of the form:
!
!  dy(x,t)    d /                          d / y(x,t) \ \
!  ------- + -- | y(x,t)*v(x) - d(x)*g(x)*-- | ------ | | = s(x)-b(x)*y(x,t)
!     dt     dx \                         dx \  g(x)  / /
!
! Where all d/dx spatial derivatives are 3-D derivatives.
!============================================================================

  
  !---------------------------------------------------------------
  !            INIT ROUTINE FOR ADVECTION-DIFFUSION
  !---------------------------------------------------------------
  subroutine advdiff_init(imetric,nx,ny,nz,timedep)
    implicit none
    integer :: imetric
    integer :: nx,ny,nz
    logical, optional :: timedep
    !
    ! Cleanup first
    !
    call advdiff_finish()
    !
    ! Copy
    !
    diff_nx      = nx
    diff_ny      = ny
    diff_nz      = nz
    diff_lda     = nx*ny*nz
    if(present(timedep)) then
       diff_timedep = timedep
    else
       diff_timedep = .false.    ! Default: Stationary
    endif
    !
    ! Check the dimension
    !
    if(nx.lt.1) then
       write(*,*) 'Error: nx.lt.1'
       stop 13
    endif
    if(ny.lt.1) then
       write(*,*) 'Error: ny.lt.1'
       stop 13
    endif
    if(nz.lt.1) then
       write(*,*) 'Error: nz.lt.1'
       stop 13
    endif
    if(nx.eq.1) then
       write(*,*) 'Sorry, x-dimension cannot be switched off'
       stop 13
    endif
    if((nz.gt.1).and.(ny.eq.1)) then
       write(*,*) 'Sorry, y-dimension cannot be switched off if '
       write(*,*) '     z-dimension switched on'
       stop 13
    endif
    if(ny.eq.1) then
       diff_idim = 1
    elseif(nz.eq.1) then
       diff_idim = 2
    else
       diff_idim = 3
    endif
    !
    ! The metric
    !          
    diff_imetric = imetric
    !
    ! Switch on/off dimensions
    !
    diff_dimy = .true.
    diff_dimz = .true.
    if(diff_idim.lt.3) diff_dimz=.false.
    if(diff_idim.lt.2) diff_dimy=.false.
    !
    ! Check array sizes
    !
    if(nx.lt.3) then
       write(*,*) 'ERROR: Array size to small in x-direction'
       stop 13
    endif
    if(diff_dimy) then
       if(ny.lt.3) then
          write(*,*) 'ERROR: Array size to small in y-direction'
          stop 13
       endif
    endif
    if(diff_dimz) then
       if(nz.lt.3) then
          write(*,*) 'ERROR: Array size to small in z-direction'
          stop 13
       endif
    endif
    !
    ! First allocate the coordinates
    !
    allocate(diff_x(nx))
    allocate(diff_y(ny))
    allocate(diff_z(nz))
    !
    ! Depending on the dimension, how many bands of the matrix do we need
    !
    if(diff_idim.eq.1) then
       diff_nband = 1
    elseif(diff_idim.eq.2) then
       diff_nband = 3
    elseif(diff_idim.eq.3) then
       diff_nband = 9
    else
       stop 111
    endif
    !
    ! In case of less than 3 dimensions, we must know at which
    ! z (and y, in case of 1-D) the equations should be solved.
    ! Also we wish to know the values of zl,zr (yl,yr) of the
    ! cell boundaries in this(ese) extra dimenion(s)
    !
    if(diff_imetric.eq.METRIC_XYZ) then
       !
       ! Cartesian [X,Y,Z]
       !
       if(.not.diff_dimy) then 
          diff_y(1) = 0.d0
          diff_yl   = -0.5d0
          diff_yr   = 0.5d0
       endif
       if(.not.diff_dimz) then
          diff_z(1) = 0.d0
          diff_zl   = -0.5d0
          diff_zr   = 0.5d0
       endif
    elseif(diff_imetric.eq.METRIC_RTHPHI) then
       !
       ! Spherical [R,Theta,Phi]
       !
       if(.not.diff_dimy) then 
          diff_y(1) = 1.5707963268d0      
          diff_yl   = 0.d0
          diff_yl   = 3.14159265359d0
       endif
       if(.not.diff_dimz) then
          diff_z(1) = 0.d0
          diff_zl   = -3.14159265359d0
          diff_zr   = 3.14159265359d0
       endif
    elseif(diff_imetric.eq.METRIC_RZPHI) then
       !
       ! Cylindrical [R,Z,Phi]
       !
       if(.not.diff_dimy) then 
          diff_y(1) = 0.d0
          diff_yl   = -0.5d0
          diff_yr   =  0.5d0
       endif
       if(.not.diff_dimz) then
          diff_z(1) = 0.d0
          diff_zl   = -3.14159265359d0
          diff_zr   = 3.14159265359d0
       endif
    elseif(diff_imetric.eq.METRIC_RZPHI_ZCONICAL) then
       !
       ! Cylindrical [R,Z,Phi], but with Z-grid (here diff_y!) proportional to r
       ! So diff_y is in fact Z/R
       !
       if(.not.diff_dimy) then 
          diff_y(1) = 0.d0
          diff_yl   = -0.5d0
          diff_yr   =  0.5d0
       endif
       if(.not.diff_dimz) then
          diff_z(1) = 0.d0
          diff_zl   = -3.14159265359d0
          diff_zr   = 3.14159265359d0
       endif
    else
       write(*,*) 'Sorry, metric type ',diff_imetric,' not known.'
       stop 13
    endif
    !
    ! Allocate arrays
    !
    allocate(diff_pxl(ny,nz))
    allocate(diff_qxl(ny,nz))
    allocate(diff_rxl(ny,nz))
    allocate(diff_pxr(ny,nz))
    allocate(diff_qxr(ny,nz))
    allocate(diff_rxr(ny,nz))
    allocate(diff_pyl(nx,nz))
    allocate(diff_qyl(nx,nz))
    allocate(diff_ryl(nx,nz))
    allocate(diff_pyr(nx,nz))
    allocate(diff_qyr(nx,nz))
    allocate(diff_ryr(nx,nz))
    allocate(diff_pzl(nx,ny))
    allocate(diff_qzl(nx,ny))
    allocate(diff_rzl(nx,ny))
    allocate(diff_pzr(nx,ny))
    allocate(diff_qzr(nx,ny))
    allocate(diff_rzr(nx,ny))
    allocate(diff_s(nx,ny,nz))
    allocate(diff_g(nx,ny,nz))
    allocate(diff_vx(nx,ny,nz))
    allocate(diff_vy(nx,ny,nz))
    allocate(diff_vz(nx,ny,nz))
    allocate(diff_dcx(nx,ny,nz))
    allocate(diff_dcy(nx,ny,nz))
    allocate(diff_dcz(nx,ny,nz))
    allocate(diff_b(nx,ny,nz))
    allocate(diff_j(nx,ny,nz))
    allocate(diff_error(nx,ny,nz))
    allocate(diff_acp(NBMAX,ZBMAX,diff_lda))
    allocate(diff_rhs(diff_lda))
    allocate(diff_sol(diff_lda))
    allocate(diff_mixb(NBMAX))
    allocate(diff_hx(nx,ny,nz))
    allocate(diff_hy(nx,ny,nz))
    allocate(diff_hz(nx,ny,nz))
    !
    ! If time-dependent, then allocate more arrays
    !
    if(diff_timedep) then
       allocate(diff_cvrho(nx,ny,nz))
    endif
    !
    ! Reset the input arrays to zero (to avoid accidential crazy values)
    !
    diff_j(:,:,:)         = 0.d0
    diff_s(:,:,:)         = 0.d0
    diff_g(:,:,:)         = 1.d0   ! Here set g=1 as default
    diff_vx(:,:,:)        = 0.d0
    diff_vy(:,:,:)        = 0.d0
    diff_vz(:,:,:)        = 0.d0
    diff_dcx(:,:,:)       = 0.d0
    diff_dcy(:,:,:)       = 0.d0
    diff_dcz(:,:,:)       = 0.d0
    diff_b(:,:,:)         = 0.d0
    diff_pxl(:,:)         = 0.d0
    diff_qxl(:,:)         = 0.d0
    diff_rxl(:,:)         = 0.d0
    diff_pxr(:,:)         = 0.d0
    diff_qxr(:,:)         = 0.d0
    diff_rxr(:,:)         = 0.d0
    diff_pyl(:,:)         = 0.d0
    diff_qyl(:,:)         = 0.d0
    diff_ryl(:,:)         = 0.d0
    diff_pyr(:,:)         = 0.d0
    diff_qyr(:,:)         = 0.d0
    diff_ryr(:,:)         = 0.d0
    diff_pzl(:,:)         = 0.d0
    diff_qzl(:,:)         = 0.d0
    diff_rzl(:,:)         = 0.d0
    diff_pzr(:,:)         = 0.d0
    diff_qzr(:,:)         = 0.d0
    diff_rzr(:,:)         = 0.d0
    diff_hx(:,:,:)        = 0.d0
    diff_hy(:,:,:)        = 0.d0
    diff_hz(:,:,:)        = 0.d0
    !
  end subroutine advdiff_init

  !---------------------------------------------------------------
  !            FINALIZE ROUTINE FOR ADVECTION-DIFFUSION
  !---------------------------------------------------------------
  subroutine advdiff_finish()
    implicit none
    if(allocated(diff_x))       deallocate(diff_x)
    if(allocated(diff_y))       deallocate(diff_y)
    if(allocated(diff_z))       deallocate(diff_z)
    if(allocated(diff_pxl))     deallocate(diff_pxl)
    if(allocated(diff_qxl))     deallocate(diff_qxl)
    if(allocated(diff_rxl))     deallocate(diff_rxl)
    if(allocated(diff_pxr))     deallocate(diff_pxr)
    if(allocated(diff_qxr))     deallocate(diff_qxr)
    if(allocated(diff_rxr))     deallocate(diff_rxr)
    if(allocated(diff_pyl))     deallocate(diff_pyl)
    if(allocated(diff_qyl))     deallocate(diff_qyl)
    if(allocated(diff_ryl))     deallocate(diff_ryl)
    if(allocated(diff_pyr))     deallocate(diff_pyr)
    if(allocated(diff_qyr))     deallocate(diff_qyr)
    if(allocated(diff_ryr))     deallocate(diff_ryr)
    if(allocated(diff_pzl))     deallocate(diff_pzl)
    if(allocated(diff_qzl))     deallocate(diff_qzl)
    if(allocated(diff_rzl))     deallocate(diff_rzl)
    if(allocated(diff_pzr))     deallocate(diff_pzr)
    if(allocated(diff_qzr))     deallocate(diff_qzr)
    if(allocated(diff_rzr))     deallocate(diff_rzr)
    if(allocated(diff_s))       deallocate(diff_s)
    if(allocated(diff_g))       deallocate(diff_g)
    if(allocated(diff_vx))      deallocate(diff_vx)
    if(allocated(diff_vy))      deallocate(diff_vy)
    if(allocated(diff_vz))      deallocate(diff_vz)
    if(allocated(diff_dcx))     deallocate(diff_dcx)
    if(allocated(diff_dcy))     deallocate(diff_dcy)
    if(allocated(diff_dcz))     deallocate(diff_dcz)
    if(allocated(diff_b))       deallocate(diff_b)
    if(allocated(diff_j))       deallocate(diff_j)
    if(allocated(diff_error))   deallocate(diff_error)
    if(allocated(diff_acp))     deallocate(diff_acp)
    if(allocated(diff_rhs))     deallocate(diff_rhs)
    if(allocated(diff_sol))     deallocate(diff_sol)
    if(allocated(diff_mixb))    deallocate(diff_mixb)
    if(allocated(diff_hx))      deallocate(diff_hx)
    if(allocated(diff_hy))      deallocate(diff_hy)
    if(allocated(diff_hz))      deallocate(diff_hz)
  end subroutine advdiff_finish
  
  !---------------------------------------------------------------
  !             THE ADVECTION DIFFUSION MAIN ROUTINE
  !
  ! First call advdiff_init() to allocate all arrays and set things
  ! up. Then insert your problem values into the following
  ! arrays (always with prefix diff_, see above):
  !
  !    x,y,z         The grid in x, y and z (for 2-D or 1-D the
  !                  z resp y,z arrays are ignored).
  !    pxl,qxl,rxl   The p,q,r of the boundary condition in 
  !                    the x-dimension, at ix=1.
  !    pxr,qxr,rxr   The p,q,r of the boundary condition in 
  !                    the x-dimension, at ix=nx.
  !    pyl,qyl,ryl   The p,q,r of the boundary condition in 
  !                    the y-dimension, at iy=1.
  !     .... etc
  !    s             The source term of the equation 
  !                    (Note: this is for fluxes H=F/4pi, so this 
  !                     source is s=q/4pi, where q is erg/s/cm^3)
  !    dcx           The diffusion coefficient in x-direction
  !    dcy           The diffusion coefficient in y-direction
  !    dcz           The diffusion coefficient in z-direction
  !    b             The beta factor before J on the rhs of the eq
  !    g             The weighting
  !    j             The initial guess for the to-be-solved J
  !
  ! Then call this subroutine.
  !
  ! ARGUMENTS:
  !
  !    convcrit      The convergence criterion (error tolerance)
  !    itermax       The maximum number of iterations
  !    savematrhs    If .true. then write the band matrix and rhs
  !                  to file (useful for debugging purposes).
  !    timestep      (for time-dependence) Time step
  !
  ! RESULT:
  !    j             The result for the mean intensity J
  !    error         The error in the flux conservation at ix,iy,iz.
  !---------------------------------------------------------------
  subroutine advdiff_solve_advectiondiffusion(convcrit,itermax,savematrhs,timestep)
    implicit none
    integer :: itermax
    doubleprecision :: convcrit
    doubleprecision :: comp(-1:1,-1:1,-1:1)
    doubleprecision :: err,dum,dum1,dum2
    doubleprecision :: dumxl,dumxr,dumyl,dumyr,dumzl,dumzr
    doubleprecision :: vol,surf_m,surf_p,norm
    doubleprecision :: eta,xx,yy,zz
    doubleprecision, optional :: timestep
    integer :: ix,iy,iz,iter,im,ixb,ixe,iyb,iye,izb,ize,iix,iiy,iiz
    integer :: nxb,nxe,nyb,nye,nzb,nze
    logical :: savematrhs
    !
    ! Check the grid for monotonicity
    !
    if(diff_x(2).gt.diff_x(1)) then
       diff_sgnx = 1.d0
    else
       diff_sgnx = -1.d0
    endif
    do ix=3,diff_nx
       if(diff_sgnx*(diff_x(ix)-diff_x(ix-1)).lt.0.d0) then
          write(*,*) 'X-grid not monotonic!'
          stop 13
       endif
    enddo
    if(diff_dimy) then
       if(diff_y(2).gt.diff_y(1)) then
          diff_sgny = 1.d0
       else
          diff_sgny = -1.d0
       endif
       do iy=3,diff_ny
          if(diff_sgny*(diff_y(iy)-diff_y(iy-1)).lt.0.d0) then
             write(*,*) 'Y-grid not monotonic!'
             stop 13
          endif
       enddo
    else
       diff_sgny = 1.d0
    endif
    if(diff_dimz) then
       if(diff_z(2).gt.diff_z(1)) then
          diff_sgnz = 1.d0
       else
          diff_sgnz = -1.d0
       endif
       do iz=3,diff_nz
          if(diff_sgnz*(diff_z(iz)-diff_z(iz-1)).lt.0.d0) then
             write(*,*) 'Z-grid not monotonic!'
             stop 13
          endif
       enddo
    else
       diff_sgnz = 1.d0
    endif
    !
    ! Check source and b
    !     
    do iz=1,diff_nz
       do iy=1,diff_ny
          do ix=1,diff_nx
             if(diff_s(ix,iy,iz).lt.0.d0) then
                write(*,*) 'ERROR in calling solve_advectiondiffusion():'
                write(*,*) '      Source contains negative'
                write(*,*) '      numbers'
                stop 13
             endif
             if(diff_b(ix,iy,iz).lt.0.d0) then
                write(*,*) 'ERROR in calling solve_advectiondiffusion():'
                write(*,*) '      Factor b contains negative'
                write(*,*) '      numbers'
                stop 13
             endif
          enddo
       enddo
    enddo
    !
    ! Now set up the matrix and the rhs
    !
    call advdiff_create_the_band_matrix_and_diff_rhs()
    !
    ! If time-dependent, then add some terms
    !
    if(diff_timedep.and.(present(timestep))) then
       call advdiff_add_time_dependence_terms(timestep)
    endif
    !
    ! For debugging: write matrix and the rhs. 
    !     
    if(savematrhs) then
       call write_band_matrix(diff_lda,diff_nband,ZBMAX)
       call write_righthandside(diff_lda)
    endif
    !
    ! Next solve the equations
    !
    call linbcg(diff_lda,diff_rhs,diff_sol,ICONVCRIT,convcrit,itermax,iter,err)
    !
    ! Store convergence information
    !
    diff_mateq_error = err
    diff_mateq_iter  = iter
    !
    ! Messages
    !
    if(.not.diff_silent) then
       if(iter.lt.itermax) then
          write(*,*) ' Convergence after ',iter,' iterations'
          write(*,*) ' Error  :',err
       else
          write(*,*) ' No convergence after ',iter,' iterations'
          write(*,*) ' Error  :',err
       endif
    endif
    !
    ! Copy things back into the arrays
    !     
    do iz=1,diff_nz
       do iy=1,diff_ny
          do ix=1,diff_nx
             !
             ! First get the eta-function
             !
             xx    = diff_x(ix)
             yy    = diff_y(iy)
             zz    = diff_z(iz)
             call eta_function(diff_imetric,xx,yy,zz,eta)
             !
             ! Then retrieve the solution
             !
             im = (ix-1) + (iy-1)*diff_nx + (iz-1)*diff_nx*diff_ny + 1
             diff_j(ix,iy,iz) = diff_sol(im) / eta
             !
          enddo
       enddo
    enddo
    !
    ! Recreate the fluxes from the solution
    !
    ! ...First set up stuff
    !
    ixb=-1
    ixe=+1
    nxb=2
    nxe=diff_nx-1
    if(diff_dimy) then
       nyb=2
       nye=diff_ny-1
       iyb=-1
       iye=+1
    else
       nyb=1
       nye=1
       iyb=0
       iye=0
    endif
    if(diff_dimz) then
       nzb=2
       nze=diff_nz-1
       izb=-1
       ize=+1
    else
       nzb=1
       nze=1
       izb=0
       ize=0
    endif
    !
    ! ...then clear arrays
    !
    do iz=1,diff_nz
       do iy=1,diff_ny
          do ix=1,diff_nx
             diff_error(ix,iy,iz) = 0.d0
             diff_hx(ix,iy,iz) = 0.d0
             if(diff_dimy) diff_hy(ix,iy,iz) = 0.d0
             if(diff_dimz) diff_hz(ix,iy,iz) = 0.d0
          enddo
       enddo
    enddo
    !
    ! ...the Hx flux
    !
    do iz=nzb,nze
       do iy=nyb,nye
          do ix=1,diff_nx-1
             call advdiff_find_components_flux(1,1,ix,iy,iz,comp)
             dum = 0.d0
             do iiz=izb,ize
                do iiy=iyb,iye
                   do iix=0,1
                      im = (ix+iix-1) + (iy+iiy-1)*diff_nx + (iz+iiz-1)*diff_nx*diff_ny + 1
                      dum=dum+comp(iix,iiy,iiz)*diff_sol(im)
                   enddo
                enddo
             enddo
             diff_hx(ix,iy,iz)=dum
          enddo
       enddo
    enddo
    !
    ! ...the Hy flux
    !
    if(diff_dimy) then
    do iz=nzb,nze
       do iy=1,diff_ny-1
          do ix=nxb,nxe
             call advdiff_find_components_flux(2,1,ix,iy,iz,comp)
             dum = 0.d0
             do iiz=izb,ize
                do iiy=0,1
                   do iix=ixb,ixe
                      im = (ix+iix-1) + (iy+iiy-1)*diff_nx + (iz+iiz-1)*diff_nx*diff_ny + 1
                      dum=dum+comp(iix,iiy,iiz)*diff_sol(im)
                   enddo
                enddo
             enddo
             diff_hy(ix,iy,iz)=dum
          enddo
       enddo
    enddo
    endif
    !
    ! ...the Hz flux
    !
    if(diff_dimz) then
    do iz=1,diff_nz-1
       do iy=nyb,nye
          do ix=nxb,nxe
             call advdiff_find_components_flux(3,1,ix,iy,iz,comp)
             dum = 0.d0
             do iiz=0,1
                do iiy=iyb,iye
                   do iix=ixb,ixe
                      im = (ix+iix-1) + (iy+iiy-1)*diff_nx + (iz+iiz-1)*diff_nx*diff_ny + 1
                      dum=dum+comp(iix,iiy,iiz)*diff_sol(im)
                   enddo
                enddo
             enddo
             diff_hz(ix,iy,iz)=dum
          enddo
       enddo
    enddo
    endif
    !
    ! Done...
    !     
  end subroutine advdiff_solve_advectiondiffusion

  !--------------------------------------------------------------
  !                    SET UP THE MAIN BAND MATRIX
  !--------------------------------------------------------------
  subroutine advdiff_create_the_band_matrix_and_diff_rhs()
    implicit none
    doubleprecision :: matelem(-1:1,-1:1,-1:1)
    doubleprecision :: p,eta,x,y,z
    integer :: im,izm,ibm
    integer :: ix,iy,iz
    integer :: ixb,ixe,iyb,iye,izb,ize
    !
    ! Then clear the matrix and boundary condition arrays
    !
    do im=1,diff_lda
       do izm=1,ZBMAX
          do ibm=1,NBMAX
             diff_acp(ibm,izm,im) = 0.d0
          enddo
       enddo
    enddo
    !
    ! Set the offset positions of the bands (measured horizontally)
    !
    diff_mixb(1) = 0                               ! The diagonal and x-bands, mid band in z
    if(diff_idim.ge.2) then
       diff_mixb(2) = - diff_nx                    ! The lower band in y-dir,  mid band in z
       diff_mixb(3) = + diff_nx                    ! The upper band in y-dir,  mid band in z
    endif
    if(diff_idim.eq.3) then
       diff_mixb(4) = 0 - diff_nx*diff_ny          ! The diagonal and x-bands, lower band in z
       diff_mixb(5) = - diff_nx - diff_nx*diff_ny  ! The lower band in y-dir,  lower band in z
       diff_mixb(6) = + diff_nx - diff_nx*diff_ny  ! The upper band in y-dir,  lower band in z
       diff_mixb(7) = 0 + diff_nx*diff_ny          ! The diagonal and x-bands, upper band in z
       diff_mixb(8) = - diff_nx + diff_nx*diff_ny  ! The lower band in y-dir,  upper band in z
       diff_mixb(9) = + diff_nx + diff_nx*diff_ny  ! The upper band in y-dir,  upper band in z
    endif
    !
    ! Now find out the boundaries within which to fill the matrix with
    ! the PDE. If a dimension is active, then we must skip the boundary
    ! cells. If not, then use index 1 simply. 
    !
    ixb = 2
    ixe = diff_nx - 1
    if(diff_dimy) then
       iyb = 2
       iye = diff_ny - 1
    else
       iyb = 1
       iye = 1
    endif
    if(diff_dimz) then
       izb = 2
       ize = diff_nz - 1
    else
       izb = 1
       ize = 1
    endif
    !
    ! Now a loop over this domain to fill the matrix with the PDE
    !
    do iz=izb,ize
       do iy=iyb,iye
          do ix=ixb,ixe
             !
             ! Find the (local) matrix elements
             !
             call advdiff_find_matrix_elements_pde(ix,iy,iz,matelem)
             !
             ! Plug them into the big matrix
             !
             im = (ix-1) + (iy-1)*diff_nx + (iz-1)*diff_nx*diff_ny + 1
             !  
             diff_acp(1,2,im) = diff_acp(1,2,im) + matelem( 0, 0, 0)
             diff_acp(1,1,im) = diff_acp(1,1,im) + matelem(-1, 0, 0)
             diff_acp(1,3,im) = diff_acp(1,3,im) + matelem(+1, 0, 0)
             if(diff_dimy) then
                diff_acp(2,2,im) = diff_acp(2,2,im) + matelem( 0,-1, 0)
                diff_acp(2,1,im) = diff_acp(2,1,im) + matelem(-1,-1, 0)
                diff_acp(2,3,im) = diff_acp(2,3,im) + matelem(+1,-1, 0)
                diff_acp(3,2,im) = diff_acp(3,2,im) + matelem( 0,+1, 0)
                diff_acp(3,1,im) = diff_acp(3,1,im) + matelem(-1,+1, 0)
                diff_acp(3,3,im) = diff_acp(3,3,im) + matelem(+1,+1, 0)
             endif
             if(diff_dimz) then
                diff_acp(4,2,im) = diff_acp(4,2,im) + matelem( 0, 0,-1)
                diff_acp(4,1,im) = diff_acp(4,1,im) + matelem(-1, 0,-1)
                diff_acp(4,3,im) = diff_acp(4,3,im) + matelem(+1, 0,-1)
                diff_acp(5,2,im) = diff_acp(5,2,im) + matelem( 0,-1,-1)
                diff_acp(5,1,im) = diff_acp(5,1,im) + matelem(-1,-1,-1)
                diff_acp(5,3,im) = diff_acp(5,3,im) + matelem(+1,-1,-1)
                diff_acp(6,2,im) = diff_acp(6,2,im) + matelem( 0,+1,-1)
                diff_acp(6,1,im) = diff_acp(6,1,im) + matelem(-1,+1,-1)
                diff_acp(6,3,im) = diff_acp(6,3,im) + matelem(+1,+1,-1)
                diff_acp(7,2,im) = diff_acp(7,2,im) + matelem( 0, 0,+1)
                diff_acp(7,1,im) = diff_acp(7,1,im) + matelem(-1, 0,+1)
                diff_acp(7,3,im) = diff_acp(7,3,im) + matelem(+1, 0,+1)
                diff_acp(8,2,im) = diff_acp(8,2,im) + matelem( 0,-1,+1)
                diff_acp(8,1,im) = diff_acp(8,1,im) + matelem(-1,-1,+1)
                diff_acp(8,3,im) = diff_acp(8,3,im) + matelem(+1,-1,+1)
                diff_acp(9,2,im) = diff_acp(9,2,im) + matelem( 0,+1,+1)
                diff_acp(9,1,im) = diff_acp(9,1,im) + matelem(-1,+1,+1)
                diff_acp(9,3,im) = diff_acp(9,3,im) + matelem(+1,+1,+1)
             endif
          enddo
       enddo
    enddo
    !
    ! Reset rhs, and install initial guess
    !     
    do iz=1,diff_nz
       do iy=1,diff_ny
          do ix=1,diff_nx
             !
             ! First get the eta-function
             !
             x     = diff_x(ix)
             y     = diff_y(iy)
             z     = diff_z(iz)
             call eta_function(diff_imetric,x,y,z,eta)
             !
             ! Then fill the initial guess
             !
             im = (ix-1) + (iy-1)*diff_nx + (iz-1)*diff_nx*diff_ny + 1
             diff_sol(im) = eta * diff_j(ix,iy,iz)
             diff_rhs(im) = 0.d0
          enddo
       enddo
    enddo
    !
    ! Install rhs (only at non-boundary gridpoints)
    !     
    do iz=izb,ize
       do iy=iyb,iye
          do ix=ixb,ixe
             im = (ix-1) + (iy-1)*diff_nx + (iz-1)*diff_nx*diff_ny + 1
             diff_rhs(im) = diff_s(ix,iy,iz)
          enddo
       enddo
    enddo
    !
    ! Now the boundary conditions
    !
    ! ... X boundaries
    !
    do iz=izb,ize
       do iy=iyb,iye
          !
          ! Left BC
          !
          ix = 1
          !
          ! First get the eta-function
          !
          x     = diff_x(ix)
          y     = diff_y(iy)
          z     = diff_z(iz)
          call eta_function(diff_imetric,x,y,z,eta)
          !
          ! Now install the BC
          !
          call advdiff_find_components_flux(1,-1,ix+1,iy,iz,matelem)
          im = (ix-1) + (iy-1)*diff_nx + (iz-1)*diff_nx*diff_ny + 1
          p  = diff_pxl(iy,iz)
          diff_acp(1,2,im)= diff_acp(1,2,im) + p*matelem(-1,0,0)
          diff_acp(1,3,im)= diff_acp(1,3,im) + p*matelem( 0,0,0)
          if(diff_dimy) then 
             diff_acp(2,2,im) = diff_acp(2,2,im) + p*matelem(-1,-1, 0)
             diff_acp(2,3,im) = diff_acp(2,3,im) + p*matelem( 0,-1, 0)
             diff_acp(3,2,im) = diff_acp(3,2,im) + p*matelem(-1,+1, 0)
             diff_acp(3,3,im) = diff_acp(3,3,im) + p*matelem( 0,+1, 0)
          endif
          if(diff_dimz) then 
             diff_acp(4,2,im) = diff_acp(4,2,im) + p*matelem(-1, 0,-1)
             diff_acp(4,3,im) = diff_acp(4,3,im) + p*matelem( 0, 0,-1)
             diff_acp(5,2,im) = diff_acp(5,2,im) + p*matelem(-1,-1,-1)
             diff_acp(5,3,im) = diff_acp(5,3,im) + p*matelem( 0,-1,-1)
             diff_acp(6,2,im) = diff_acp(6,2,im) + p*matelem(-1,+1,-1)
             diff_acp(6,3,im) = diff_acp(6,3,im) + p*matelem( 0,+1,-1)
             diff_acp(7,2,im) = diff_acp(7,2,im) + p*matelem(-1, 0,+1)
             diff_acp(7,3,im) = diff_acp(7,3,im) + p*matelem( 0, 0,+1)
             diff_acp(8,2,im) = diff_acp(8,2,im) + p*matelem(-1,-1,+1)
             diff_acp(8,3,im) = diff_acp(8,3,im) + p*matelem( 0,-1,+1)
             diff_acp(9,2,im) = diff_acp(9,2,im) + p*matelem(-1,+1,+1)
             diff_acp(9,3,im) = diff_acp(9,3,im) + p*matelem( 0,+1,+1)
          endif
          diff_acp(1,2,im)=diff_acp(1,2,im)+diff_qxl(iy,iz)/eta
          diff_rhs(im)    =diff_rhs(im)    +diff_rxl(iy,iz)
          !
          ! Right BC
          !     
          ix = diff_nx
          !
          ! First get the eta-function
          !
          x     = diff_x(ix)
          y     = diff_y(iy)
          z     = diff_z(iz)
          call eta_function(diff_imetric,x,y,z,eta)
          !
          ! Now install the BC
          !
          call advdiff_find_components_flux(1,+1,ix-1,iy,iz,matelem)
          im = (ix-1) + (iy-1)*diff_nx + (iz-1)*diff_nx*diff_ny + 1
          p  = diff_pxr(iy,iz)
          diff_acp(1,2,im)= diff_acp(1,2,im) + p*matelem(+1, 0, 0)
          diff_acp(1,1,im)= diff_acp(1,1,im) + p*matelem( 0, 0, 0)
          if(diff_dimy) then 
             diff_acp(2,2,im) = diff_acp(2,2,im) + p*matelem(+1,-1, 0)
             diff_acp(2,1,im) = diff_acp(2,1,im) + p*matelem( 0,-1, 0)
             diff_acp(3,2,im) = diff_acp(3,2,im) + p*matelem(+1,+1, 0)
             diff_acp(3,1,im) = diff_acp(3,1,im) + p*matelem( 0,+1, 0)
          endif
          if(diff_dimz) then 
             diff_acp(4,2,im) = diff_acp(4,2,im) + p*matelem(+1, 0,-1)
             diff_acp(4,1,im) = diff_acp(4,1,im) + p*matelem( 0, 0,-1)
             diff_acp(5,2,im) = diff_acp(5,2,im) + p*matelem(+1,-1,-1)
             diff_acp(5,1,im) = diff_acp(5,1,im) + p*matelem( 0,-1,-1)
             diff_acp(6,2,im) = diff_acp(6,2,im) + p*matelem(+1,+1,-1)
             diff_acp(6,1,im) = diff_acp(6,1,im) + p*matelem( 0,+1,-1)
             diff_acp(7,2,im) = diff_acp(7,2,im) + p*matelem(+1, 0,+1)
             diff_acp(7,1,im) = diff_acp(7,1,im) + p*matelem( 0, 0,+1)
             diff_acp(8,2,im) = diff_acp(8,2,im) + p*matelem(+1,-1,+1)
             diff_acp(8,1,im) = diff_acp(8,1,im) + p*matelem( 0,-1,+1)
             diff_acp(9,2,im) = diff_acp(9,2,im) + p*matelem(+1,+1,+1)
             diff_acp(9,1,im) = diff_acp(9,1,im) + p*matelem( 0,+1,+1)
          endif
          diff_acp(1,2,im)= diff_acp(1,2,im) + diff_qxr(iy,iz)/eta
          diff_rhs(im)    = diff_rhs(im)     + diff_rxr(iy,iz)
       enddo
    enddo
    !
    ! ... Y boundaries
    !
    if(diff_dimy) then
    do iz=izb,ize
       do ix=ixb,ixe
          !
          ! Left BC
          !              
          iy = 1
          !
          ! First get the eta-function
          !
          x     = diff_x(ix)
          y     = diff_y(iy)
          z     = diff_z(iz)
          call eta_function(diff_imetric,x,y,z,eta)
          !
          ! Now install the BC
          !
          call advdiff_find_components_flux(2,-1,ix,iy+1,iz,matelem)
          im = (ix-1) + (iy-1)*diff_nx + (iz-1)*diff_nx*diff_ny + 1
          p  = diff_pyl(ix,iz)
          diff_acp(1,2,im) = diff_acp(1,2,im) + p*matelem( 0,-1, 0)
          diff_acp(3,2,im) = diff_acp(3,2,im) + p*matelem( 0, 0, 0)
          diff_acp(1,1,im) = diff_acp(1,1,im) + p*matelem(-1,-1, 0)
          diff_acp(3,1,im) = diff_acp(3,1,im) + p*matelem(-1, 0, 0)
          diff_acp(1,3,im) = diff_acp(1,3,im) + p*matelem(+1,-1, 0)
          diff_acp(3,3,im) = diff_acp(3,3,im) + p*matelem(+1, 0, 0)
          if(diff_dimz) then 
             diff_acp(4,2,im) = diff_acp(4,2,im) + p*matelem( 0,-1,-1)
             diff_acp(6,2,im) = diff_acp(6,2,im) + p*matelem( 0, 0,-1)
             diff_acp(4,1,im) = diff_acp(4,1,im) + p*matelem(-1,-1,-1)
             diff_acp(6,1,im) = diff_acp(6,1,im) + p*matelem(-1, 0,-1)
             diff_acp(4,3,im) = diff_acp(4,3,im) + p*matelem(+1,-1,-1)
             diff_acp(6,3,im) = diff_acp(6,3,im) + p*matelem(+1, 0,-1)
             diff_acp(7,2,im) = diff_acp(7,2,im) + p*matelem( 0,-1,+1)
             diff_acp(9,2,im) = diff_acp(9,2,im) + p*matelem( 0, 0,+1)
             diff_acp(7,1,im) = diff_acp(7,1,im) + p*matelem(-1,-1,+1)
             diff_acp(9,1,im) = diff_acp(9,1,im) + p*matelem(-1, 0,+1)
             diff_acp(7,3,im) = diff_acp(7,3,im) + p*matelem(+1,-1,+1)
             diff_acp(9,3,im) = diff_acp(9,3,im) + p*matelem(+1, 0,+1)
          endif
          diff_acp(1,2,im)= diff_acp(1,2,im) + diff_qyl(ix,iz)/eta
          diff_rhs(im)    = diff_rhs(im)     + diff_ryl(ix,iz)
          !
          ! Right BC
          !
          iy = diff_ny
          !
          ! First get the eta-function
          !
          x     = diff_x(ix)
          y     = diff_y(iy)
          z     = diff_z(iz)
          call eta_function(diff_imetric,x,y,z,eta)
          !
          ! Now install the BC
          !
          call advdiff_find_components_flux(2,+1,ix,iy-1,iz,matelem)
          im = (ix-1) + (iy-1)*diff_nx + (iz-1)*diff_nx*diff_ny + 1
          p  = diff_pyr(ix,iz)
          diff_acp(1,2,im) = diff_acp(1,2,im) + p*matelem( 0,+1, 0)
          diff_acp(2,2,im) = diff_acp(2,2,im) + p*matelem( 0, 0, 0)
          diff_acp(1,1,im) = diff_acp(1,1,im) + p*matelem(-1,+1, 0)
          diff_acp(2,1,im) = diff_acp(2,1,im) + p*matelem(-1, 0, 0)
          diff_acp(1,3,im) = diff_acp(1,3,im) + p*matelem(+1,+1, 0)
          diff_acp(2,3,im) = diff_acp(2,3,im) + p*matelem(+1, 0, 0)
          if(diff_dimz) then 
             diff_acp(4,2,im) = diff_acp(4,2,im) + p*matelem( 0,+1,-1)
             diff_acp(5,2,im) = diff_acp(5,2,im) + p*matelem( 0, 0,-1)
             diff_acp(4,1,im) = diff_acp(4,1,im) + p*matelem(-1,+1,-1)
             diff_acp(5,1,im) = diff_acp(5,1,im) + p*matelem(-1, 0,-1)
             diff_acp(4,3,im) = diff_acp(4,3,im) + p*matelem(+1,+1,-1)
             diff_acp(5,3,im) = diff_acp(5,3,im) + p*matelem(+1, 0,-1)
             diff_acp(7,2,im) = diff_acp(7,2,im) + p*matelem( 0,+1,+1)
             diff_acp(8,2,im) = diff_acp(8,2,im) + p*matelem( 0, 0,+1)
             diff_acp(7,1,im) = diff_acp(7,1,im) + p*matelem(-1,+1,+1)
             diff_acp(8,1,im) = diff_acp(8,1,im) + p*matelem(-1, 0,+1)
             diff_acp(7,3,im) = diff_acp(7,3,im) + p*matelem(+1,+1,+1)
             diff_acp(8,3,im) = diff_acp(8,3,im) + p*matelem(+1, 0,+1)
          endif
          diff_acp(1,2,im)= diff_acp(1,2,im) + diff_qyr(ix,iz)/eta
          diff_rhs(im)    = diff_rhs(im)     + diff_ryr(ix,iz)
       enddo
    enddo
    endif
    !
    ! ... Z boundaries
    ! 
    !        >>>> NOT YET COMPLETELY READY <<<<
    !
    if(diff_dimz) then
    write(*,*) 'WARNING: Z-BOUNDARY CONDITION STILL PRIMITIVE!!!'
    write(*,*) 'For serious use: complete the code at this point!'
    do iy=iyb,iye
       do ix=ixb,ixe
          !
          ! Left BC
          !
          iz = 1
          !
          ! First get the eta-function
          !
          x     = diff_x(ix)
          y     = diff_y(iy)
          z     = diff_z(iz)
          call eta_function(diff_imetric,x,y,z,eta)
          !
          ! Now install the BC
          !
          call advdiff_find_components_flux(3,-1,ix,iy,iz+1,matelem)
          im = (ix-1) + (iy-1)*diff_nx + (iz-1)*diff_nx*diff_ny + 1
          p  = diff_pzl(ix,iy)
          diff_acp(1,2,im)= diff_acp(1,2,im) + p*matelem(0,0,-1)
          diff_acp(8,2,im)= diff_acp(8,2,im) + p*matelem(0,0, 0)
          diff_acp(1,2,im)= diff_acp(1,2,im) + diff_qzl(ix,iy)/eta
          diff_rhs(im)    = diff_rhs(im)     + diff_rzl(ix,iy)
          !
          ! Right BC
          !
          iz = diff_nz
          !
          ! First get the eta-function
          !
          x     = diff_x(ix)
          y     = diff_y(iy)
          z     = diff_z(iz)
          call eta_function(diff_imetric,x,y,z,eta)
          !
          ! Now install the BC
          !
          call advdiff_find_components_flux(3,+1,ix,iy,iz-1,matelem)
          im = (ix-1) + (iy-1)*diff_nx + (iz-1)*diff_nx*diff_ny + 1
          p  = diff_pzr(ix,iy)
          diff_acp(1,2,im)= diff_acp(1,2,im) + p*matelem(0,0,+1)
          diff_acp(5,2,im)= diff_acp(5,2,im) + p*matelem(0,0, 0)
          diff_acp(1,2,im)= diff_acp(1,2,im) + diff_qzr(ix,iy)/eta
          diff_rhs(im)    = diff_rhs(im)     + diff_rzr(ix,iy)
       enddo
    enddo
    endif
    !
    ! Now the ribbons: average
    !
    if(diff_dimy) then
       do iz=izb,ize
          ix = 1
          iy = 1
          im = (ix-1) + (iy-1)*diff_nx + (iz-1)*diff_nx*diff_ny + 1
          diff_acp(1,2,im) = 2.d0
          diff_acp(1,3,im) = -1.0d0
          diff_acp(3,2,im) = -1.0d0
          ix = 1
          iy = diff_ny
          im = (ix-1) + (iy-1)*diff_nx + (iz-1)*diff_nx*diff_ny + 1
          diff_acp(1,2,im) = 2.d0
          diff_acp(1,3,im) = -1.d0
          diff_acp(2,2,im) = -1.d0
          ix = diff_nx
          iy = 1
          im = (ix-1) + (iy-1)*diff_nx + (iz-1)*diff_nx*diff_ny + 1
          diff_acp(1,2,im) = 2.d0
          diff_acp(1,1,im) = -1.d0
          diff_acp(3,2,im) = -1.d0
          ix = diff_nx
          iy = diff_ny
          im = (ix-1) + (iy-1)*diff_nx + (iz-1)*diff_nx*diff_ny + 1
          diff_acp(1,2,im) = 2.d0
          diff_acp(1,1,im) = -1.d0
          diff_acp(2,2,im) = -1.d0
       enddo
    endif
    if(diff_dimz) then
       do ix=ixb,ixe
          iy = 1
          iz = 1
          im = (ix-1) + (iy-1)*diff_nx + (iz-1)*diff_nx*diff_ny + 1
          diff_acp(1,2,im) = 2.d0
          diff_acp(3,2,im) = -1.d0
          diff_acp(7,2,im) = -1.d0
          iy = diff_ny
          iz = 1
          im = (ix-1) + (iy-1)*diff_nx + (iz-1)*diff_nx*diff_ny + 1
          diff_acp(1,2,im) = 2.d0
          diff_acp(2,2,im) = -1.d0
          diff_acp(7,2,im) = -1.d0
          iy = 1
          iz = diff_nz
          im = (ix-1) + (iy-1)*diff_nx + (iz-1)*diff_nx*diff_ny + 1
          diff_acp(1,2,im) = 2.d0
          diff_acp(3,2,im) = -1.d0
          diff_acp(4,2,im) = -1.d0
          iy = diff_ny
          iz = diff_nz
          im = (ix-1) + (iy-1)*diff_nx + (iz-1)*diff_nx*diff_ny + 1
          diff_acp(1,2,im) = 2.d0
          diff_acp(2,2,im) = -1.d0
          diff_acp(4,2,im) = -1.d0
       enddo
       do iy=iyb,iye
          ix = 1
          iz = 1
          im = (ix-1) + (iy-1)*diff_nx + (iz-1)*diff_nx*diff_ny + 1
          diff_acp(1,2,im) = 2.d0
          diff_acp(1,3,im) = -1.d0
          diff_acp(7,2,im) = -1.d0              
          ix = diff_nx
          iz = 1
          im = (ix-1) + (iy-1)*diff_nx + (iz-1)*diff_nx*diff_ny + 1
          diff_acp(1,2,im) = 2.d0
          diff_acp(1,1,im) = -1.d0
          diff_acp(7,2,im) = -1.d0              
          ix = 1
          iz = diff_nz
          im = (ix-1) + (iy-1)*diff_nx + (iz-1)*diff_nx*diff_ny + 1
          diff_acp(1,2,im) = 2.d0
          diff_acp(1,3,im) = -1.d0
          diff_acp(4,2,im) = -1.d0              
          ix = diff_nx
          iz = diff_nz
          im = (ix-1) + (iy-1)*diff_nx + (iz-1)*diff_nx*diff_ny + 1
          diff_acp(1,2,im) = 2.d0
          diff_acp(1,1,im) = -1.d0
          diff_acp(4,2,im) = -1.d0              
       enddo
    endif
    !
    ! Now the corners: average
    !
    if(diff_dimz) then
       ix = 1
       iy = 1
       iz = 1
       diff_acp(1,2,im) =  3.d0
       diff_acp(1,3,im) = -1.d0
       diff_acp(3,2,im) = -1.d0
       diff_acp(7,2,im) = -1.d0
       ix = diff_nx
       iy = 1
       iz = 1
       diff_acp(1,2,im) =  3.d0
       diff_acp(1,1,im) = -1.d0
       diff_acp(3,2,im) = -1.d0
       diff_acp(7,2,im) = -1.d0
       ix = 1
       iy = diff_ny
       iz = 1
       diff_acp(1,2,im) =  3.d0
       diff_acp(1,3,im) = -1.d0
       diff_acp(2,2,im) = -1.d0
       diff_acp(7,2,im) = -1.d0
       ix = 1
       iy = 1
       iz = diff_nz
       diff_acp(1,2,im) =  3.d0
       diff_acp(1,3,im) = -1.d0
       diff_acp(3,2,im) = -1.d0
       diff_acp(4,2,im) = -1.d0
       ix = diff_nx
       iy = diff_ny
       iz = 1
       diff_acp(1,2,im) =  3.d0
       diff_acp(1,1,im) = -1.d0
       diff_acp(2,2,im) = -1.d0
       diff_acp(7,2,im) = -1.d0
       ix = diff_nx
       iy = 1
       iz = diff_nz
       diff_acp(1,2,im) =  3.d0
       diff_acp(1,1,im) = -1.d0
       diff_acp(3,2,im) = -1.d0
       diff_acp(4,2,im) = -1.d0
       ix = diff_nx
       iy = diff_ny
       iz = diff_nz
       diff_acp(1,2,im) =  3.d0
       diff_acp(1,1,im) = -1.d0
       diff_acp(2,2,im) = -1.d0
       diff_acp(4,2,im) = -1.d0
    endif
    !
    ! Now normalize the rows of matrix and vector, which is in fact
    ! a very simple kind of preconditioning
    !
    call normalize_rows(diff_lda,diff_nband,3)
    !
    ! Done...
    !
  end subroutine advdiff_create_the_band_matrix_and_diff_rhs


  !---------------------------------------------------------------
  !           COMPUTE MATRIX ELEMENTS FOR DIFFUSION EQ
  !
  ! It will compute the matrix elements corresponding to the
  ! following equation:
  !
  !   (H^x*S^x)_ix+1/2,iy,iz - (H^x*S^x)_ix-1/2,iy,iz +
  !   (H^y*S^y)_ix,iy+1/2,iz - (H^y*S^y)_ix,iy-1/2,iz +
  !   (H^z*S^z)_iz,iy,iz+1/2 - (H^z*S^z)_ix,iy,iz-1/2
  !     = V_ix,iy,iz * ( s - beta*J )
  !
  !---------------------------------------------------------------
  subroutine advdiff_find_matrix_elements_pde(ix,iy,iz,matelem)
    implicit none
    integer :: ix,iy,iz
    doubleprecision :: matelem(-1:1,-1:1,-1:1)
    doubleprecision :: comp_p(-1:1,-1:1,-1:1)
    doubleprecision :: comp_m(-1:1,-1:1,-1:1)
    doubleprecision :: vol,surf_p,surf_m
    doubleprecision :: x,y,z,eta
    integer :: k,l,m
    !
    ! First checks on the indices
    !
    if((ix.lt.2).or.(ix.gt.diff_nx-1)) stop 61
    if(diff_idim.eq.3) then
        if((iz.lt.2).or.(iz.gt.diff_nz-1)) stop 63
    else
        if(iz.ne.1) stop 73
    endif
    if(diff_idim.ge.2) then
        if((iy.lt.2).or.(iy.gt.diff_ny-1)) stop 62
    else
        if(iy.ne.1) stop 72
    endif
    !
    ! First put everything to zero
    !
    do k=-1,1
       do l=-1,1
          do m=-1,1
             matelem(k,l,m) = 0.d0
          enddo
       enddo
    enddo
    !
    ! Get the volume element
    !      
    vol = get_volume_element(ix,iy,iz)
    !     
    ! Flux differences in X-direction
    !
    call advdiff_find_components_flux(1,1,ix,iy,iz,comp_p)
    call advdiff_find_components_flux(1,-1,ix,iy,iz,comp_m)
    surf_p = get_surface_element(1,2*ix+1,2*iy,2*iz)
    surf_m = get_surface_element(1,2*ix-1,2*iy,2*iz)
    do k=-1,1
       do l=-1,1
          do m=-1,1
             matelem(k,l,m) = matelem(k,l,m) + comp_p(k,l,m) * surf_p * diff_sgnx
             matelem(k,l,m) = matelem(k,l,m) - comp_m(k,l,m) * surf_m * diff_sgnx
          enddo
       enddo
    enddo
    !
    ! Flux differences in Y-direction
    !
    if(diff_dimy) then
    call advdiff_find_components_flux(2,1,ix,iy,iz,comp_p)
    call advdiff_find_components_flux(2,-1,ix,iy,iz,comp_m)
    surf_p = get_surface_element(2,2*ix,2*iy+1,2*iz)
    surf_m = get_surface_element(2,2*ix,2*iy-1,2*iz)
    do k=-1,1
       do l=-1,1
          do m=-1,1
             matelem(k,l,m) = matelem(k,l,m) + comp_p(k,l,m) * surf_p * diff_sgny
             matelem(k,l,m) = matelem(k,l,m) - comp_m(k,l,m) * surf_m * diff_sgny
          enddo
       enddo
    enddo
    endif
    !
    ! Flux differences in Z-direction
    !
    if(diff_dimz) then
    call advdiff_find_components_flux(3,1,ix,iy,iz,comp_p)
    call advdiff_find_components_flux(3,-1,ix,iy,iz,comp_m)
    surf_p = get_surface_element(3,2*ix,2*iy,2*iz+1)
    surf_m = get_surface_element(3,2*ix,2*iy,2*iz-1)
    do k=-1,1
       do l=-1,1
          do m=-1,1
             matelem(k,l,m) = matelem(k,l,m) + comp_p(k,l,m) * surf_p * diff_sgnz
             matelem(k,l,m) = matelem(k,l,m) - comp_m(k,l,m) * surf_m * diff_sgnz
          enddo
       enddo
    enddo
    endif
    !
    ! Get the eta-function
    !
    x     = diff_x(ix)
    y     = diff_y(iy)
    z     = diff_z(iz)
    call eta_function(diff_imetric,x,y,z,eta)
    !
    ! Now the source term part that is proportional to J
    !     
    matelem(0,0,0) = matelem(0,0,0) + diff_b(ix,iy,iz)*vol/eta
    !
    ! Now divide by volume of the cell
    !
    do k=-1,1
       do l=-1,1
          do m=-1,1
             matelem(k,l,m) = matelem(k,l,m) / vol
          enddo
       enddo
    enddo
  end subroutine advdiff_find_matrix_elements_pde

  !---------------------------------------------------------------
  !     FIND THE COMPONENTS FOR THE ADVECTION-DIFFUSION FLUX
  !---------------------------------------------------------------
  subroutine advdiff_find_components_flux(idir,iplus,ix,iy,iz,comp)
    implicit none
    integer :: idir,iplus,ix,iy,iz
    doubleprecision :: comp(-1:1,-1:1,-1:1)
    doubleprecision :: dx,dy,dz,x,y,z
    doubleprecision :: diffcoef,velo,sqgii,gm,gp,g,dummy,diffcoefr
    integer :: ix2,iy2,iz2,ip,im,k,l,m
    !
    ! Some useful stuff
    !
    if(iplus.eq.1) then
        ip = 1
        im = 0
    elseif(iplus.eq.-1) then
        ip = 0
        im = -1
    else
        stop 76
    endif
    !
    ! Get the half-integers
    !
    if(idir.eq.1) then
        ix2 = 2*ix + iplus
        iy2 = 2*iy
        iz2 = 2*iz
    elseif(idir.eq.2) then
        ix2 = 2*ix
        iy2 = 2*iy + iplus
        iz2 = 2*iz
    elseif(idir.eq.3) then
        ix2 = 2*ix
        iy2 = 2*iy
        iz2 = 2*iz + iplus
    else
        stop 57
    endif
    !
    ! Get the velocity
    !
    velo     = advdiff_get_velocity(idir,ix2,iy2,iz2)
    !
    ! Get the diffusion coefficient
    !
    diffcoef = advdiff_get_diffcoef(idir,ix2,iy2,iz2)
    !
    ! Get the g coefficients (as in g*d(y/g)/dx)
    !
    if(idir.eq.1) then
       gm    = diff_g(ix+im,iy,iz)
       gp    = diff_g(ix+ip,iy,iz)
    elseif(idir.eq.2) then
       gm    = diff_g(ix,iy+im,iz)
       gp    = diff_g(ix,iy+ip,iz)
    elseif(idir.eq.3) then
       gm    = diff_g(ix,iy,iz+im)
       gp    = diff_g(ix,iy,iz+ip)
    else
       stop 82
    endif
    g = 0.5 * ( gm + gp )
    !
    ! Get the dx, dy and dz
    !
    dx = get_dgrid(1,ix2)
    if(diff_dimy) then
        dy = get_dgrid(2,iy2) 
    else 
        dy = 0.d0
    endif
    if(diff_dimz) then
        dz = get_dgrid(3,iz2) 
    else
        dz = 0.d0
    endif
    !
    ! First put everything to zero
    !
    do k=-1,1
        do l=-1,1
            do m=-1,1
                comp(k,l,m) = 0.d0
            enddo
        enddo
    enddo
    !
    ! Do the main stuff
    !
    if(idir.eq.1) then
       !
       ! Get the sqrt(g_11)
       !
       x     = 0.5d0 * ( diff_x(ix+ip) + diff_x(ix+im) )
       y     = diff_y(iy)
       z     = diff_z(iz)
       sqgii = sqroot_gij(diff_imetric,idir,x,y,z)
       !
       ! Add the advection term
       !
       if(velo.ge.0.0) then
          comp(im,0,0)  = comp(im,0,0)  + velo
       else
          comp(ip,0,0)  = comp(ip,0,0)  + velo
       endif
       !
       ! Add the derivative for the diffusion
       !
       comp(ip,0,0)  = comp(ip,0,0)  - g * diffcoef / dx / sqgii / gp
       comp(im,0,0)  = comp(im,0,0)  + g * diffcoef / dx / sqgii / gm
       !
       ! For the METRIC_RZPHI_ZCONICAL, we also have cross terms
       !
       if(diff_dimy.and.(diff_imetric.eq.METRIC_RZPHI_ZCONICAL)) then
          dummy         = 0.5d0 * g * diffcoef * (y/x) / dy
          comp(ip,+1,0) = comp(ip,+1,0) - dummy / diff_g(ix+ip,iy+1,iz)
          comp(im,+1,0) = comp(im,+1,0) - dummy / diff_g(ix+im,iy+1,iz)
          comp(ip,-1,0) = comp(ip,-1,0) + dummy / diff_g(ix+ip,iy-1,iz)
          comp(im,-1,0) = comp(im,-1,0) + dummy / diff_g(ix+im,iy-1,iz)
       endif
       !
    elseif(idir.eq.2) then
       !
       ! Get the sqrt(g_22)
       !
       x     = diff_x(ix)
       y     = 0.5d0 * ( diff_y(iy+ip) + diff_y(iy+im) )
       z     = diff_z(iz)
       sqgii = sqroot_gij(diff_imetric,idir,x,y,z)
       !
       ! Add the advection term
       !
       if(velo.ge.0.0) then
          comp(0,im,0)  = comp(0,im,0)  + velo
       else
          comp(0,ip,0)  = comp(0,ip,0)  + velo
       endif
       !
       ! For the METRIC_RZPHI_ZCONICAL the diffusion coefficient in
       ! z-direction is slightly enhanced
       !
       if(diff_imetric.eq.METRIC_RZPHI_ZCONICAL) then
          diffcoefr = advdiff_get_diffcoef(1,ix2,iy2,iz2)
          diffcoef  = diffcoef + (y**2) * diffcoefr
       endif
       !
       ! Add the derivative for the diffusion
       !
       comp(0,ip,0)  = comp(0,ip,0)  - g * diffcoef / dy / sqgii / gp
       comp(0,im,0)  = comp(0,im,0)  + g * diffcoef / dy / sqgii / gm
       !
       ! For the METRIC_RZPHI_ZCONICAL, we also have cross terms
       !
       if(diff_imetric.eq.METRIC_RZPHI_ZCONICAL) then
          dummy         = 0.5d0 * g * diffcoefr * y / dx
          comp(+1,ip,0) = comp(+1,ip,0) - dummy / diff_g(ix+1,iy+ip,iz)
          comp(+1,im,0) = comp(+1,im,0) - dummy / diff_g(ix+1,iy+im,iz)
          comp(-1,ip,0) = comp(-1,ip,0) + dummy / diff_g(ix-1,iy+ip,iz)
          comp(-1,im,0) = comp(-1,im,0) + dummy / diff_g(ix-1,iy+im,iz)
       endif
       !
    elseif(idir.eq.3) then
       !
       ! Get the sqrt(g_33)
       !
       x     = diff_x(ix)
       y     = diff_y(iy)
       z     = 0.5d0 * ( diff_z(iz+ip) + diff_z(iz+im) )
       sqgii = sqroot_gij(diff_imetric,idir,x,y,z)
       !
       ! Add the advection term
       !
       if(velo.ge.0.0) then
          comp(0,0,im)  = comp(0,0,im)  + velo
       else
          comp(0,0,ip)  = comp(0,0,ip)  + velo
       endif
       !
       ! Add the derivative for the diffusion
       !
       comp(0,0,ip)  = comp(0,0,ip)  - g * diffcoef / dz / sqgii / gp
       comp(0,0,im)  = comp(0,0,im)  + g * diffcoef / dz / sqgii / gm
       !
    else
       stop 57
    endif
    !
  end subroutine advdiff_find_components_flux

  !---------------------------------------------------------------
  !         FIND THE DIFFUSION COEFFICIENT AT CELL SURFACE
  !
  ! ARGUMENTS:
  !     idir        = Direction of diffcoef: dc_x, or _y or _z?
  !     ix2,iy2,iz2 = Twice the grid position, plus 0 or 1 when
  !                      in cell center ix2=2*(ix) or cell boundary 
  !                      ix2=2*(ix+1/2). Note that for e.g. idir.eq.1
  !                      the ix2 must be odd while the iy2 must be
  !                      even.
  !
  !---------------------------------------------------------------
  function advdiff_get_diffcoef(idir,ix2,iy2,iz2)
    implicit none
    integer :: ix2,iy2,iz2,ix,iy,iz,ixp,iyp,izp,idir
    doubleprecision :: advdiff_get_diffcoef
    !
    ! Find out which cell boundary
    !
    ix=ix2/2
    iy=iy2/2
    iz=iz2/2
    if(.not.even(ix2)) then
       ixp=ix2/2+1
    else
       ixp=ix2/2
    endif
    if(.not.even(iy2)) then
       iyp=iy2/2+1
    else
       iyp=iy2/2
    endif
    if(.not.even(iz2)) then
       izp=iz2/2+1
    else
       izp=iz2/2
    endif
    !
    ! If out of bounds, then error
    !
    if((ix2.ge.2*diff_nx+1).or.(iy2.ge.2*diff_ny+1).or.   &
       (iz2.ge.2*diff_nz+1).or.(ix2.le.1).or.(iy2.le.1).or.(iz2.le.1)) then
       write(*,*) 'Internal error: ix2,iy2,iz2 out of bounds'
       write(*,*) 'ix2=',ix2,' iy2=',iy2,' iz2=',iz2
       write(*,*) 2*diff_nx+1,2*diff_ny+1,2*diff_nz+1
       stop 15
    endif
    !
    ! Return the diffusion coefficient
    !
    if(idir.eq.1) then
       advdiff_get_diffcoef = 0.5d0 * ( diff_dcx(ix,iy,iz) + diff_dcx(ixp,iyp,izp) )
    elseif(idir.eq.2) then
       advdiff_get_diffcoef = 0.5d0 * ( diff_dcy(ix,iy,iz) + diff_dcy(ixp,iyp,izp) )
    elseif(idir.eq.3) then
       advdiff_get_diffcoef = 0.5d0 * ( diff_dcz(ix,iy,iz) + diff_dcz(ixp,iyp,izp) )
    else
       stop 923
    endif
    !
    return
  end function advdiff_get_diffcoef

  !---------------------------------------------------------------
  !                FIND THE VELOCITY AT CELL SURFACE
  !
  ! ARGUMENTS:
  !     idir        = Direction of velocity: v_x, or _y or _z?
  !     ix2,iy2,iz2 = Twice the grid position, plus 0 or 1 when
  !                      in cell center ix2=2*(ix) or cell boundary 
  !                      ix2=2*(ix+1/2). Note that for e.g. idir.eq.1
  !                      the ix2 must be odd while the iy2 must be
  !                      even.
  !
  !---------------------------------------------------------------
  function advdiff_get_velocity(idir,ix2,iy2,iz2)
    implicit none
    integer :: ix2,iy2,iz2,ix,iy,iz,ixp,iyp,izp,idir
    doubleprecision :: advdiff_get_velocity,zeta
    !
    ! Find out which cell boundary
    !
    ix=ix2/2
    iy=iy2/2
    iz=iz2/2
    if(.not.even(ix2)) then
       ixp=ix2/2+1
    else
       ixp=ix2/2
    endif
    if(.not.even(iy2)) then
       iyp=iy2/2+1
    else
       iyp=iy2/2
    endif
    if(.not.even(iz2)) then
       izp=iz2/2+1
    else
       izp=iz2/2
    endif
    !
    ! If out of bounds, then error
    !
    if((ix2.ge.2*diff_nx+1).or.(iy2.ge.2*diff_ny+1).or.   &
       (iz2.ge.2*diff_nz+1).or.(ix2.le.1).or.(iy2.le.1).or.(iz2.le.1)) then
       write(*,*) 'Internal error: ix2,iy2,iz2 out of bounds'
       write(*,*) 'ix2=',ix2,' iy2=',iy2,' iz2=',iz2
       write(*,*) 2*diff_nx+1,2*diff_ny+1,2*diff_nz+1
       stop 15
    endif
    !
    ! Return the velocity
    !
    if(idir.eq.1) then
       advdiff_get_velocity = 0.5d0 * ( diff_vx(ix,iy,iz) + diff_vx(ixp,iyp,izp) )
    elseif(idir.eq.2) then
       advdiff_get_velocity = 0.5d0 * ( diff_vy(ix,iy,iz) + diff_vy(ixp,iyp,izp) )
       if(diff_imetric.eq.METRIC_RZPHI_ZCONICAL) then
          zeta                 = 0.5d0 * ( diff_y(iy) + diff_y(iyp) )
          advdiff_get_velocity = advdiff_get_velocity - &
               zeta * 0.5d0 * ( diff_vx(ix,iy,iz) + diff_vx(ixp,iyp,izp) )
       endif
    elseif(idir.eq.3) then
       advdiff_get_velocity = 0.5d0 * ( diff_vz(ix,iy,iz) + diff_vz(ixp,iyp,izp) )
    else
       stop 923
    endif
    !
    return
  end function advdiff_get_velocity

  !---------------------------------------------------------------
  !         MATRIX-ELEMENTS AND RHS FOR TIME-DEPENDENCE
  !---------------------------------------------------------------
  subroutine advdiff_add_time_dependence_terms(timestep)
    implicit none
    double precision :: timestep,dtinverse
    integer :: ix,iy,iz
    integer :: ixb,ixe,iyb,iye,izb,ize,im
    !
    ! First find out the boundaries within which to fill the matrix with
    ! the PDE. If a dimension is active, then we must skip the boundary
    ! cells. If not, then use index 1 simply. 
    !
    ixb = 2
    ixe = diff_nx - 1
    if(diff_dimy) then
       iyb = 2
       iye = diff_ny - 1
    else
       iyb = 1
       iye = 1
    endif
    if(diff_dimz) then
       izb = 2
       ize = diff_nz - 1
    else
       izb = 1
       ize = 1
    endif
    !
    ! Compute the inverse time step
    !
    dtinverse = 1.d0 / timestep
    !
    ! Now a loop over the domain, apart from the boundaries, to add
    ! the terms for time-dependence to the matrix and rhs
    !
    do iz=izb,ize
       do iy=iyb,iye
          do ix=ixb,ixe
             im = (ix-1) + (iy-1)*diff_nx + (iz-1)*diff_nx*diff_ny + 1
             !
             ! Add 1/dt to matrix elements
             !
             diff_acp(1,2,im) = diff_acp(1,2,im) + dtinverse
             !
             ! Add (1/dt)*J to rhs
             !
             diff_rhs(im) = diff_rhs(im) + dtinverse * diff_j(ix,iy,iz)
          enddo
       enddo
    enddo
    !
  end subroutine advdiff_add_time_dependence_terms
 


  !===============================================================
  !       ANALYTIC (EXACT) FORMULAE FOR GEOMETRIC QUANTITIES
  !===============================================================


  !---------------------------------------------------------------
  !                     THE ETA-FUNCTION
  !
  ! This may be any function of x,y,z. The best thing would be
  ! to make it a function that makes eta*J a not very strongly
  ! varying function of x,y,z. For spherical coordinates this
  ! would be for instance eta=R^2. Here I simply take the
  ! functions I believe are best. For (R,theta,phi) I take R^2,
  ! for (R,Z,phi) I take R and for (x,y,z) I take 1. 
  !---------------------------------------------------------------
  subroutine eta_function(imetric,x,y,z,eta)
    implicit none
    !
    integer :: imetric
    doubleprecision :: x,y,z,eta
    !
    if(imetric.eq.METRIC_XYZ) then
       !
       ! Cartesian [X,Y,Z]
       !
       eta      = 1.d0
       !
    elseif(imetric.eq.METRIC_RTHPHI) then
       !
       ! Spherical [R,Theta,Phi]
       !
       ! We can try with
       !
       eta      = 1.d0
       !
       ! If that does not work (e.g. for very large Rout/Rin 
       ! ratio), try:
       !
       !  eta      = x**2
       !
    elseif(imetric.eq.METRIC_RZPHI) then
       !
       ! Cylindrical [R,Z,Phi]
       !
       ! We can try with
       !
       eta      = 1.d0
       !
       ! If that does not work (e.g. for very large Rout/Rin 
       ! ratio), try:
       !
       !  eta      = x
       !
    elseif(imetric.eq.METRIC_RZPHI_ZCONICAL) then
       eta      = 1.d0
    else
       stop 57
    endif
    !
  end subroutine eta_function
  

  !---------------------------------------------------------------
  !         ANALYTIC FORMULAE FOR SQUARE ROOT OF METRIC
  !
  ! Note that the metric must be diagonal for this module.
  !---------------------------------------------------------------
  function sqroot_gij(imetric,idir,x,y,z)
    implicit none
    integer :: imetric,idir
    doubleprecision :: x,y,z,e,sqroot_gij
    !
    if(imetric.eq.METRIC_XYZ) then
       !
       ! Cartesian [X,Y,Z]
       !
       if(idir.eq.1) then
          e = 1.d0
       elseif(idir.eq.2) then
          e = 1.d0
       elseif(idir.eq.3) then
          e = 1.d0
       else
          stop 55
       endif
    elseif(imetric.eq.METRIC_RTHPHI) then
       !
       ! Spherical [R,Theta,Phi]
       !
       if(idir.eq.1) then
          e = 1.d0
       elseif(idir.eq.2) then
          e = x
       elseif(idir.eq.3) then
          e = x * sin(y)
       else
          stop 55
       endif
    elseif(imetric.eq.METRIC_RZPHI) then
       !
       ! Cylindrical [R,Z,Phi]
       !
       if(idir.eq.1) then
          e = 1.d0
       elseif(idir.eq.2) then
          e = 1.d0
       elseif(idir.eq.3) then
          e = x
       else
          stop 55
       endif
    elseif(imetric.eq.METRIC_RZPHI_ZCONICAL) then
       !
       ! Cylindrical [R,Z,Phi], but with Z-grid (here diff_y!) proportional to r
       ! So diff_y is in fact Z/R
       !
       if(idir.eq.1) then
          e = 1.d0
       elseif(idir.eq.2) then
          e = x
       elseif(idir.eq.3) then
          e = x
       else
          stop 55
       endif
    else
       stop 57
    endif
    !
    sqroot_gij = e
    return
  end function sqroot_gij


  !---------------------------------------------------------------
  !            ANALYTIC FORMULAE FOR SURFACE ELEMENTS
  !---------------------------------------------------------------
  function surface_element(imetric,idir,x1,x2,y1,y2,z1,z2)
    implicit none
    integer :: imetric,idir
    doubleprecision :: x1,x2,y1,y2,z1,z2,surface_element,s
    !
    if(imetric.eq.METRIC_XYZ) then
       !
       ! Cartesian [X,Y,Z]
       !
       if(idir.eq.1) then
          s = (y2-y1) * (z2-z1)
       elseif(idir.eq.2) then
          s = (x2-x1) * (z2-z1)
       elseif(idir.eq.3) then
          s = (x2-x1) * (y2-y1)
       else
          stop 55
       endif
    elseif(imetric.eq.METRIC_RTHPHI) then
       !
       ! Spherical [R,Theta,Phi]
       !
       if(idir.eq.1) then
          s = x1**2 * ( cos(y1) - cos(y2) ) * ( z2 - z1 )
       elseif(idir.eq.2) then
          s = 0.5d0 * ( x2**2 - x1**2 ) * sin(y1) * ( z2 - z1 )
       elseif(idir.eq.3) then
          s = 0.5d0 * ( x2**2 - x1**2 ) * ( y2 - y1 ) 
       else
          stop 55
       endif
    elseif(imetric.eq.METRIC_RZPHI) then
       !
       ! Cylindrical [R,Z,Phi]
       !
       if(idir.eq.1) then
          s = x1 * ( y2 - y1 ) * ( z2 - z1 )
       elseif(idir.eq.2) then
          s = 0.5d0 * ( x2**2 - x1**2 ) * ( z2 - z1 )
       elseif(idir.eq.3) then
          s = ( x2 - x1 ) * ( y2 - y1 )
       else
          stop 55
       endif
    elseif(imetric.eq.METRIC_RZPHI_ZCONICAL) then
       !
       ! Cylindrical [R,Z,Phi], but with Z-grid (here diff_y!) proportional to r
       ! So diff_y is in fact Z/R
       !
       ! NOTE: The surface in Z-direction (i.e. in diff_y, i.e. idir.eq.2) should
       !       be the vertically _projected_ surface, because the z-flux is pointing
       !       vertically (and not perpendicularly to the conical surface).
       !
       if(idir.eq.1) then
          s = x1**2 * ( y2 - y1 ) * ( z2 - z1 )
       elseif(idir.eq.2) then
          s = 0.5d0 * ( x2**2 - x1**2 ) * ( z2 - z1 )
       elseif(idir.eq.3) then
          s = 0.5d0 * ( x2**2 - x1**2 ) * ( y2 - y1 )
       else
          stop 55
       endif
    else
       stop 57
    endif
    surface_element = abs(s)
    return
  end function surface_element


  !---------------------------------------------------------------
  !           ANALYTIC FORMULAE FOR VOLUME ELEMENTS
  !---------------------------------------------------------------
  function volume_element(imetric,x1,x2,y1,y2,z1,z2)
    implicit none
    integer :: imetric
    doubleprecision :: x1,x2,y1,y2,z1,z2,volume_element,v
    !
    if(imetric.eq.METRIC_XYZ) then
       !
       ! Cartesian [X,Y,Z]
       !
       v = ( x2 - x1 ) * ( y2 - y1 ) * ( z2 - z1 ) 
       !
    elseif(imetric.eq.METRIC_RTHPHI) then
       !
       ! Spherical [R,Theta,Phi]
       !
       v = (1.d0/3.d0) * ( x2**3 - x1**3 ) * ( cos(y1) - cos(y2) ) * ( z2 - z1 )
       !
    elseif(imetric.eq.METRIC_RZPHI) then
       !
       ! Cylindrical [R,Z,Phi]
       !
       v = 0.5d0 * ( x2**2 - x1**2 ) * ( y2 - y1 ) * ( z2 - z1 )
       !
    elseif(imetric.eq.METRIC_RZPHI_ZCONICAL) then
       !
       ! Cylindrical [R,Z,Phi], but with Z-grid (here diff_y!) proportional to r
       ! So diff_y is in fact Z/R
       !
       v =  (1.d0/3.d0) * ( x2**2 - x1**2 ) * ( y2 - y1 ) * ( z2 - z1 )
       !
    else
       stop 57
    endif
    volume_element = abs(v)
    return
  end function volume_element
  

  !---------------------------------------------------------------
  !            ANALYTIC FORMULAE FOR CONNECTION
  !
  ! Since the covariant flux H^i is a covariant divergence of 
  ! the (f^ij*J) tensor, we have
  !
  !    - alpha H^i = nabla_j(f^ij*J) 
  !
  !        = partial_j(f^ij*J) +
  !               [ Gamma^j_jk*f^ik + Gamma^i_jk*f^kj ] * J
  !
  ! The factor between [] is computed here in this function, for
  ! a given index i (=idir).
  !
  ! Note: In the earlier version of this code we included the
  !       eta factor derivatives here as well. In the present
  !       version we instead put eta directly in f^ij: we 
  !       write f^ij*J = (f^ij/eta)*(eta*J) = (f^ij/eta)*sol
  !       where sol is the solution vector of the matrix 
  !       equation. 
  !---------------------------------------------------------------
  function connectionfactor(imetric,idir,x,y,z,fxx,fxy,fyy,fxz,fyz,fzz)
    implicit none
    integer :: imetric,idir
    doubleprecision :: connectionfactor
    doubleprecision :: x,y,z,fxx,fxy,fyy,fxz,fyz,fzz
    doubleprecision :: c,cossint,sint
    !
    ! First the geometric stuff
    !
    if(imetric.eq.METRIC_XYZ) then
       !
       ! Cartesian [X,Y,Z]
       !
       if(idir.eq.1) then
          c = 0.d0
       elseif(idir.eq.2) then
          c = 0.d0
       elseif(idir.eq.3) then
          c = 0.d0
       else
          stop 55
       endif
    elseif(imetric.eq.METRIC_RTHPHI) then
       !
       ! Spherical [R,Theta,Phi]
       !
       sint = sin(y)
       if(sint.eq.0.d0) then
          write(*,*) 'Error: at Theta=0,pi equations diverge'
          stop 13
       endif
       cossint  = cos(y) / sint
       if(idir.eq.1) then
          c = (2.d0/x)*fxx + cossint*fxy - x*(fyy+fzz*sint**2) 
       elseif(idir.eq.2) then
          c = (4.d0/x)*fxy + cossint*(fyy-fzz*sint**2)
       elseif(idir.eq.3) then
          c = (4.d0/x)*fxz + 3.d0*cossint*fyz
       else
          stop 55
       endif
    elseif(imetric.eq.METRIC_RZPHI) then
       !
       ! Cylindrical [R,Z,Phi]
       !
       !    >> Check these once more (22-08-00) <<
       !
       if(idir.eq.1) then
          c = (1.d0/x)*fxx - x*fzz
       elseif(idir.eq.2) then
          c = (1.d0/x)*fxy
       elseif(idir.eq.3) then
          c = (3.d0/x)*fxz
       else
          stop 55
       endif
    else
       stop 57
    endif
    !
    connectionfactor = c
    return
  end function connectionfactor
  

  !--------------------------------------------------------------
  !                      IS A NUMBER EVEN?   
  !--------------------------------------------------------------
  function even(i)
    implicit none
    integer i
    logical even
    !
    if(i/2.eq.(i+1)/2) then
       even = .true.
    else
       even = .false.
    endif
    !
    return
  end function even
  


  !==============================================================
  !          ROUTINES FOR THE MATRIX EQUATION SOLVER
  !==============================================================


  !-----------------------------------------------------------
  !          HELPER ROUTINE FOR LINBCG : MATRIX MULT
  !-----------------------------------------------------------
  subroutine atimes(n,x,r,itrnsp)
    integer :: n,itrnsp
    doubleprecision :: x(n),r(n)
    if (itrnsp.eq.0) then
       call bmat_mult_vec(n,x,r)
    else
       call bmat_trans_mult_vec(n,x,r)
    endif
    return
  end subroutine atimes
      

  !-----------------------------------------------------------
  !          HELPER ROUTINE FOR LINBCG : MATRIX PRECOND
  !-----------------------------------------------------------
  subroutine asolve(n,b,x,itrnsp)
    integer :: n,itrnsp,i
    double precision :: x(n),b(n)
    do i=1,n
       x(i)=b(i)/diff_acp(1,2,i)
    enddo
    return
  end subroutine asolve
      

  !------------------------------------------------------------
  ! MULTIPLY MATRIX WITH VECTOR
  !
  !  This routine multiplies a multiband matrix with a vector.
  !
  !    diff_acp(1..nb,1..zb,1..n)   The multiband matrix
  !    diff_mixb(1..nb)             The offset vertical pos of each 
  !                            band. Start counting from 
  !                            leftmost band element;
  !                            i.e diagonal has mixb=1 and zb=1.
  !                            a tridiag matrix has mixb=0
  !                            and zb=3.
  !    vecin(1..n)             The vector to me multiplied
  !    vecout(1..n)            The answer vector
  !
  !  One should specify the dimensions of the matrix:
  !
  !    nb = Number of bands (typically 3 for 2-D diffusion prblm)
  !    zb = Number of rows per band on each column (typical 3
  !          times the number of equations to solve)
  !    n  = Number of columns of the matrix
  !
  !  and the routine will compute:
  !
  !     VECOUT := ACP * VECIN 
  !------------------------------------------------------------
  subroutine bmat_mult_vec(n,vecin,vecout)
    implicit none
    doubleprecision :: vecin(n)
    doubleprecision :: vecout(n)
    integer :: n
    integer :: i,j,k,ixxb
    !
    ! Clear the vecout
    !
    do j=1,n
       vecout(j) = 0.d0
    enddo
    !      
    ! Do the multiplication band-by-band
    !
    do i=1,diff_nband
       !
       ixxb = diff_mixb(i)
       !
       ! Loop over rows. Skip rows for which the band is outside
       ! of the matrix limits.
       !
       do j=max(1,3-ZBMAX-ixxb),min(n,n+1-ixxb)
          !
          ! Multiply this row with the vector, and make sure
          ! to skip parts of the band that are outside the
          ! scope of the matrix
          !             
          do k=max(1,j+ixxb-1),min(n,j+ixxb+ZBMAX-2)
             vecout(j) = vecout(j) + diff_acp(i,k-j-ixxb+2,j) * vecin(k)
          enddo
          !
       enddo
    enddo
    !      
    return
  end subroutine bmat_mult_vec


  !------------------------------------------------------------
  ! MULTIPLY TRANSPOSE OF MATRIX WITH VECTOR
  !
  ! See bmat_mult_vec() above, but then for transpose of matrix.
  !------------------------------------------------------------
  subroutine bmat_trans_mult_vec(n,vecin,vecout)
    implicit none
    doubleprecision :: vecin(n)
    doubleprecision :: vecout(n)
    integer :: n
    integer :: i,j,k,ixxb
    !
    ! Clear the vecout
    !
    do j=1,n
       vecout(j) = 0.d0
    enddo
    !      
    ! Do the multiplication band-by-band
    !
    do i=1,diff_nband
       !
       ixxb = diff_mixb(i)
       !
       ! Loop over rows. Skip rows for which the band is outside
       ! of the matrix limits.
       !
       do j=max(1,3-ZBMAX-ixxb),min(n,n+1-ixxb)
          !
          ! Multiply this row with the vector, and make sure
          ! to skip parts of the band that are outside the
          ! scope of the matrix
          !             
          do k=max(1,j+ixxb-1),min(n,j+ixxb+ZBMAX-2)
             vecout(k) = vecout(k) + diff_acp(i,k-j-ixxb+2,j) * vecin(j)
          enddo
          !
       enddo
    enddo
    !      
    return
  end subroutine bmat_trans_mult_vec


  !------------------------------------------------------------
  !                     NORMALIZE ROWS
  ! 
  ! This routine is meant to multiply the rows of the matrix
  ! and the rhs with a number to make things more normalized.
  !------------------------------------------------------------
  subroutine normalize_rows(nn,nb,nz)
    implicit none
    integer :: nn,nb,nz
    doubleprecision :: dummy  
    integer :: in,ib,iz
    !
    do in=1,nn
       dummy = 0.d0
       do ib=1,nb
          do iz=1,nz
             dummy = dummy + abs(diff_acp(ib,iz,in))
          enddo
       enddo
       dummy = 6.d0 / dummy
       diff_rhs(in) = diff_rhs(in) * dummy
       do ib=1,nb
          do iz=1,nz
             diff_acp(ib,iz,in) = diff_acp(ib,iz,in) * dummy
          enddo
       enddo
    enddo
    !
  end subroutine normalize_rows


  !-----------------------------------------------------------
  !        WRITE THE BANDS OF THE MATRIX TO A FILE
  !-----------------------------------------------------------
  subroutine write_band_matrix(nn,nb,nz)
    implicit none
    integer :: nn,nb,nz
    integer :: ib,iz,in
    !
    open(unit=1,file='bandmatrix.dat',status='unknown')
    write(1,*) nn,nb,nz
    do in=1,nn
       do ib=1,nb
          do iz=1,nz
             write(1,*) diff_acp(ib,iz,in)
          enddo
       enddo
       write(1,*)
    enddo
    close(1)
    !
  end subroutine write_band_matrix


  !-----------------------------------------------------------
  !        WRITE THE RIGHT HAND SIDE OF THE EQUATION
  !-----------------------------------------------------------
  subroutine write_righthandside(nn)
    implicit none
    integer :: nn
    integer :: in
    !
    open(unit=1,file='righthandside.dat',status='unknown')
    write(1,*) nn
    do in=1,nn
       write(1,*) diff_rhs(in)
    enddo
    close(1)
    !
  end subroutine write_righthandside


  !========================================================================
  !                    NUMERICAL RECIPES ROUTINES
  !========================================================================
  
  subroutine linbcg(n,b,x,itol,tol,itmax,iter,err)
    integer :: iter,itmax,itol,n
    double precision :: err,tol,b(n),x(n)
    double precision, parameter :: eps = 1.d-14
    integer :: j
    double precision :: ak,akden,bk,bkden,bknum,bnrm,dxnrm,xnrm,zm1nrm
    double precision :: znrm,p(n),pp(n),r(n),rr(n),z(n),zz(n)
    iter=0
    call atimes(n,x,r,0)
    do j=1,n
       r(j)=b(j)-r(j)
       rr(j)=r(j)
    enddo
    !     call atimes(n,r,rr,0)
    znrm=1.d0
    if(itol.eq.1) then
       bnrm=snrm(n,b,itol)
    else if (itol.eq.2) then
       call asolve(n,b,z,0)
       bnrm=snrm(n,z,itol)
    else if (itol.eq.3.or.itol.eq.4) then
       call asolve(n,b,z,0)
       bnrm=snrm(n,z,itol)
       call asolve(n,r,z,0)
       znrm=snrm(n,z,itol)
    else
       stop 'illegal itol in linbcg'
    endif
    call asolve(n,r,z,0)
  100 if (iter.le.itmax) then
       iter=iter+1
       zm1nrm=znrm
       call asolve(n,rr,zz,1)
       bknum=0.d0
       do j=1,n
          bknum=bknum+z(j)*rr(j)
       enddo
       if(iter.eq.1) then
          do j=1,n
             p(j)=z(j)
             pp(j)=zz(j)
          enddo
       else
          bk=bknum/bkden
          do j=1,n
             p(j)=bk*p(j)+z(j)
             pp(j)=bk*pp(j)+zz(j)
          enddo
       endif
       bkden=bknum
       call atimes(n,p,z,0)
       akden=0.d0
       do j=1,n
          akden=akden+z(j)*pp(j)
       enddo
       ak=bknum/akden
       call atimes(n,pp,zz,1)
       do j=1,n
          x(j)=x(j)+ak*p(j)
          r(j)=r(j)-ak*z(j)
          rr(j)=rr(j)-ak*zz(j)
       enddo
       call asolve(n,r,z,0)
       if(itol.eq.1.or.itol.eq.2)then
          znrm=1.d0
          err=snrm(n,r,itol)/bnrm
       else if(itol.eq.3.or.itol.eq.4)then
          znrm=snrm(n,z,itol)
          if(abs(zm1nrm-znrm).gt.EPS*znrm) then
             dxnrm=abs(ak)*snrm(n,p,itol)
             err=znrm/abs(zm1nrm-znrm)*dxnrm
          else
             err=znrm/bnrm
             goto 100
          endif
          xnrm=snrm(n,x,itol)
          if(err.le.0.5d0*xnrm) then
             err=err/xnrm
          else
             err=znrm/bnrm
             goto 100
          endif
       endif
       !write (*,*) ' iter=',iter,' err=',err
       if(err.gt.tol) goto 100
    endif
    return
  end subroutine linbcg
  !  (C) Copr. 1986-92 Numerical Recipes Software =v1.9"217..
  
  
  function snrm(n,sx,itol)
    integer :: n,itol,i,isamax
    double precision :: sx(n),snrm
    if (itol.le.3)then
       snrm=0.
       do i=1,n
          snrm=snrm+sx(i)**2
       enddo
       snrm=sqrt(snrm)
    else
       isamax=1
       do i=1,n
          if(abs(sx(i)).gt.abs(sx(isamax))) isamax=i
       enddo
       snrm=abs(sx(isamax))
    endif
    return
  end function snrm
  !  (C) Copr. 1986-92 Numerical Recipes Software =v1.9"217..
  
end module diffusion
