import numpy as np
from . import diffusion


def solvediff3d(geom, x, y, z, diffcoef, source, boundcond, jinit=None,
                bcoef=None, gcoef=None, convcrit=1e-10, itermax=10000, dt=0,
                retflux=False,silent=True):
    """
    Solve a 1D, 2D or 3D diffusion problem, either stationary (dt==0) or time-dependent
    for a single time step (dt>0).

    NOTE: This subroutine uses the diffusion.f90 Fortran-90 module written by
          C.P. Dullemond. This module has to be compiled with

            f2py -c diffusion.f90 -m diffusion

          This will create diffusion.so, which is the module used here.

    It solves for J the following equation:

          dJ
        g -- - Div ( D Grad J )  =  S - b*J
          dt

    The boundary conditions are either Dirichlet, in which case the value of J is set at the
    boundary, or it follows the following formula:

        p n.H + q J  = r

    where n is the outward-pointing normal vector, H is the flux vector given by

        H = - D Grad J

    and p, q and r are values to be given by the user through the boundcond list (see below).
    Note that for the left, right boundary we have, respectively (for x as an example):

        left:   n.H = - H_x         right:   n.H = + H_x        (likewise for y and z)

    Arguments:
    ----------

     geom : str
        'cartesian'   = Cartesian coordinates
        'spherical'   = Spherical coordinates with x=r, y=theta and z=phi
        'cylindrical' = Cylindrical coordinates with x=r, y=z and z=phi

     x : array
        values of x. If just one value: dimension is not active.

     y : array
        values of y. If just one value: dimension is not active.

     z : array
        values of z. If just one value: dimension is not active.

     diffcoef : array
        Array of values of the diffusion coefficient D(x,y,z)

     source : array
        Array of values of the source term S(x,y,z)

     boundcond : list of lists
        [bcx,bcy,bcz].
        Example bcx (boundary conditions in x direction): bcx is a list of
        boundary conditions, one for left, one for right: bcx = [bcxl,bcxr].
        Examples for left:
        bcxl = ['dirichlet',value] : The left x boundary is a Dirichlet
                                   boundary condition with J=value. The
                                   value can be either a number, or an array
                                   of numbers (in this case an ny,nz array).
        bcxl = ['flux',p,q,r] :    More general bc: p n.Flux + q J  = r, where
                                   Flux = D Grad J, and n is the outward-pointing
                                   normal vector.

     jinit : array
        Array of initial guess of J (or value of J at start of time step): J_init(x,y,z)

     bcoef : array
        Array of values of the coefficient b(x,y,z) (If None, then taken to be 0)

     gcoef : array
        Array of values of the coefficient g(x,y,z) (only for time-dependent) (If None, then taken to be 1)

     convcrit : float
        Convergence criterion to be used for the linear equation solver.

     itermax : int
        Maximum number of convergence steps

     dt : float
        Time step for time-dependent case. If dt=0, then stationary case.

     retflux : bool
        If True, then not only J but also Hx, Hy, Hz are returned
    """
    #
    # First make sure that any pre-existing arrays in the Fortran-90 module are deallocated
    #
    diffusion.diffusion.diff_finish()
    #
    # Interpret the geometry
    #
    if geom == 'cartesian':
        imetric = 1
    elif geom == 'spherical':
        imetric = 2
    elif geom == 'cylindrical':
        imetric = 3
        raise ValueError('For now, cylindrical geometry is still under construction (the diffusion.f90 still has to be changed to r,phi,z instead of r,z,phi).')
    else:
        raise ValueError('Geometry not known')
    #
    # Interpret time dependency
    #
    if dt == 0:
        timedep = False
    else:
        timedep = True
        assert jinit is not None, "For time-dependent diffusion we need an initial J"
    #
    # Find length of all dimensions
    #
    nx = len(x)
    ny = len(y)
    nz = len(z)
    assert nx > 0, 'Each dimension must have at least one value of its coordinate'
    assert ny > 0, 'Each dimension must have at least one value of its coordinate'
    assert nz > 0, 'Each dimension must have at least one value of its coordinate'
    nn = [nx, ny, nz]
    #
    # Initialize the diffusion.f90 module, meaning that the internal arrays will be
    # allocated.
    #
    diffusion.diffusion.diff_init(imetric, nx, ny, nz, timedep)
    diffusion.diffusion.diff_silent = silent
    #
    # Now copy the grid into the arrays or the diffusion.90 module
    #
    diffusion.diffusion.diff_x[:] = x[:]
    diffusion.diffusion.diff_y[:] = y[:]
    diffusion.diffusion.diff_z[:] = z[:]
    #
    # Set the traceless Eddington factors to zero
    #
    diffusion.diffusion.diff_wxx[:] = 0.0
    diffusion.diffusion.diff_wyy[:] = 0.0
    diffusion.diffusion.diff_wxy[:] = 0.0
    diffusion.diffusion.diff_wxz[:] = 0.0
    diffusion.diffusion.diff_wyz[:] = 0.0
    #
    # Install the diffusion coefficient
    #
    # NOTE: The factor 1/3 is because of the Eddington factor
    #       being 1/3 at high optical depth.
    #
    diffusion.diffusion.diff_ax[:] = (1.0/3.0) / diffcoef[:]
    diffusion.diffusion.diff_ay[:] = diffusion.diffusion.diff_ax[:]
    diffusion.diffusion.diff_az[:] = diffusion.diffusion.diff_ax[:]
    #
    # Install the source
    #
    diffusion.diffusion.diff_s[:] = source[:]
    #
    # If the initial guess (or value) for J (jinit)
    # Note: This can be either jinit (standard case) or temp
    #
    if jinit is not None:
        diffusion.diffusion.diff_j[:] = jinit[:]
    else:
        diffusion.diffusion.diff_j[:] = 0.0
    #
    # If bcoef is set, then install
    #
    if bcoef is not None:
        diffusion.diffusion.diff_b[:] = bcoef[:]
    else:
        diffusion.diffusion.diff_b[:] = 0.0
    #
    # If time-dependent diffusion equation
    #
    if timedep:
        #
        # If gcoef is set, then install
        #
        if gcoef is not None:
            diffusion.diffusion.diff_gfact[:] = gcoef[:]
        else:
            diffusion.diffusion.diff_gfact[:] = 1.0
        diffusion.diffusion.diff_cvrho[:] = 0.0    # We do any c_v*rho stuff for radiative transfer in Python
    #
    # Set all other arrays from diffusion.f90 to zero
    #
    diffusion.diffusion.diff_t_irrad[:] = 0.0  # We do any irradiation stuff for radiative transfer in Python
    diffusion.diffusion.diff_tgas[:] = 0.0     # We do any gas temperature for radiative transfer in Python
    #
    # Now handle the boundary conditions
    #
    pp = [[diffusion.diffusion.diff_pxl, diffusion.diffusion.diff_pxr],
          [diffusion.diffusion.diff_pyl, diffusion.diffusion.diff_pyr],
          [diffusion.diffusion.diff_pzl, diffusion.diffusion.diff_pzr]]
    qq = [[diffusion.diffusion.diff_qxl, diffusion.diffusion.diff_qxr],
          [diffusion.diffusion.diff_qyl, diffusion.diffusion.diff_qyr],
          [diffusion.diffusion.diff_qzl, diffusion.diffusion.diff_qzr]]
    rr = [[diffusion.diffusion.diff_rxl, diffusion.diffusion.diff_rxr],
          [diffusion.diffusion.diff_ryl, diffusion.diffusion.diff_ryr],
          [diffusion.diffusion.diff_rzl, diffusion.diffusion.diff_rzr]]
    sgn = [-1, 1]
    for idir in range(3):
        if nn[idir] > 1:
            for ilr in range(2):
                bc = boundcond[idir][ilr]
                p  = pp[idir][ilr]
                q  = qq[idir][ilr]
                r  = rr[idir][ilr]
                if bc[0] == 'dirichlet':
                    p[:] = 0.0
                    q[:] = 1.0
                    r[:] = bc[1]
                elif bc[0] == 'flux':
                    p[:] = sgn[ilr] * bc[1]   # Because diffusion.f90 does not have the n.H, but H.
                    q[:] = bc[2]
                    r[:] = bc[3]
                else:
                    raise ValueError('Do not know boundary condition type')
    #
    # Now solve
    #
    if timedep:
        diffusion.diffusion.solve_diffusion(convcrit, itermax, False, timestep=dt)
    else:
        diffusion.diffusion.solve_diffusion(convcrit, itermax, False)
    #
    # If error is too large, then warn
    #
    if(diffusion.diffusion.diff_mateq_error > convcrit):
        print("Warning: Matrix equation solver did not converge. Error remains: {}".format(diffusion.diffusion.diff_mateq_error))
    # print("  Nr of iterations of matrix equation solver: {}".format(diffusion.diffusion.diff_mateq_iter))
    #
    # Copy the result
    #
    jnew = diffusion.diffusion.diff_j.copy()
    #
    # If the user wants, we compute also the H-fluxes
    #
    # Note: The hflux[][:,:,:] arrays are actually the *interface*
    #       fluxes: The hflux[0][0,1,1] is the flux between
    #       cell [0,1,1] and [1,1,1], and likewise for the other
    #       directions. hflux[0] should actually have dimension
    #       hflux[0][nx-1,ny,nz] (and likewise for other directions),
    #       but all hflux[:][:,:,:] have [nx,ny,nz] dimension.
    #       Consequently hflux[0][-1,:,:] = 0.0 (unused elements),
    #       and likewise hflux[1][:,-1,:] = 0.0 and
    #       hflux[2][:,:,-1] = 0.0.
    #
    if retflux:
        hflux = [None, None, None]
        if nx > 1:
            hflux[0] = diffusion.diffusion.diff_hx.copy()
        if ny > 1:
            hflux[1] = diffusion.diffusion.diff_hy.copy()
        if nz > 1:
            hflux[2] = diffusion.diffusion.diff_hz.copy()
    #
    # Finish the diffusion module, freeing all its memory
    #
    diffusion.diffusion.diff_finish()
    #
    # Return the result
    #
    # If the user wants, we return also the H-fluxes
    #
    if retflux:
        return jnew, hflux
    else:
        return jnew

def solveadvdiff3d(geom, x, y, z, vx, vy, vz, diffcoef, source, boundcond, jinit=None,
                bcoef=None, wcoef=None, convcrit=1e-10, itermax=10000, dt=0,
                   retflux=False,silent=True):
    """
    Solve a 1D, 2D or 3D diffusion problem, either stationary (dt==0) or time-dependent
    for a single time step (dt>0).

    NOTE: This subroutine uses the diffusion.f90 Fortran-90 module written by
          C.P. Dullemond. This module has to be compiled with

            f2py -c diffusion.f90 -m diffusion

          This will create diffusion.so, which is the module used here.

    It solves for J the following equation:

          dJ
          -- + Div ( J v - w D Grad (J/w) )  =  S - b*J
          dt

    The boundary conditions are either Dirichlet, in which case the value of J is set at the
    boundary, or it follows the following formula:

        p n.F + q J  = r

    where n is the outward-pointing normal vector, F is the flux vector given by

        F = - D Grad J

    and p, q and r are values to be given by the user through the boundcond list (see below).
    Note that for the left, right boundary we have, respectively (for x as an example):

        left:   n.F = - F_x         right:   n.F = + F_x        (likewise for y and z)

    Arguments:
    ----------

     geom : str
        'cartesian'           = Cartesian coordinates
        'spherical'           = Spherical coordinates with x=r, y=theta and z=phi
        'cylindrical'         = Cylindrical coordinates with x=r, y=z   and z=phi
        'cylindricalzconical' = Cylindrical coordinates with x=r, y=z/r and z=phi

        WARNING: In Cylindrical coordinates the y is the vertical (i.e. 'z') coordinate,
                 while the z is the azimuthal coordinate. Can be confusing, but the logic
                 is that 2-D can only be done properly in r,z, and the active dimensions
                 should be the first two. 

     x : array
        values of x. If just one value: dimension is not active.

     y : array
        values of y. If just one value: dimension is not active.

     z : array
        values of z. If just one value: dimension is not active.

     vx, vy, vz : array
        Array of values of the x, y and z velocities (cell-centered)

     diffcoef : array
        Array of values of the diffusion coefficient D(x,y,z) (cell-centered)

     source : array
        Array of values of the source term S(x,y,z)

     boundcond : list of lists
        [bcx,bcy,bcz].
        Example bcx (boundary conditions in x direction): bcx is a list of
        boundary conditions, one for left, one for right: bcx = [bcxl,bcxr].
        Examples for left:
        bcxl = ['dirichlet',value] : The left x boundary is a Dirichlet
                                   boundary condition with J=value. The
                                   value can be either a number, or an array
                                   of numbers (in this case an ny,nz array).
        bcxl = ['flux',p,q,r] :    More general bc: p n.Flux + q J  = r, where
                                   Flux = D Grad J, and n is the outward-pointing
                                   normal vector.

     jinit : array
        Array of initial guess of J (or value of J at start of time step): J_init(x,y,z)

     bcoef : array
        Array of values of the coefficient b(x,y,z) (If None, then taken to be 0)

     wcoef : array
        Array of values of the coefficient w(x,y,z) (If None, then taken to be 0)

     convcrit : float
        Convergence criterion to be used for the linear equation solver.

     itermax : int
        Maximum number of convergence steps

     dt : float
        Time step for time-dependent case. If dt=0, then stationary case.

     retflux : bool
        If True, then not only J but also the fluxes Fx, Fy, Fz are returned
    """
    #
    # First make sure that any pre-existing arrays in the Fortran-90 module are deallocated
    #
    diffusion.diffusion.advdiff_finish()
    #
    # Interpret the geometry
    #
    if geom == 'cartesian':
        imetric = 1
    elif geom == 'spherical':
        imetric = 2
    elif geom == 'cylindrical':
        imetric = 3
    elif geom == 'cylindricalzconical':
        imetric = 4
    else:
        raise ValueError('Geometry not known')
    #
    # Interpret time dependency
    #
    if dt == 0:
        timedep = False
    else:
        timedep = True
        assert jinit is not None, "For time-dependent diffusion we need an initial J"
    #
    # Find length of all dimensions
    #
    nx = len(x)
    ny = len(y)
    nz = len(z)
    assert nx > 0, 'Each dimension must have at least one value of its coordinate'
    assert ny > 0, 'Each dimension must have at least one value of its coordinate'
    assert nz > 0, 'Each dimension must have at least one value of its coordinate'
    nn = [nx, ny, nz]
    #
    # Initialize the diffusion.f90 module, meaning that the internal arrays will be
    # allocated.
    #
    diffusion.diffusion.advdiff_init(imetric, nx, ny, nz, timedep)
    diffusion.diffusion.diff_silent = silent
    #
    # Now copy the grid into the arrays or the diffusion.90 module
    #
    diffusion.diffusion.diff_x[:] = x[:]
    diffusion.diffusion.diff_y[:] = y[:]
    diffusion.diffusion.diff_z[:] = z[:]
    #
    # Install the velocities
    #
    diffusion.diffusion.diff_vx[:]   = vx[:]
    diffusion.diffusion.diff_vy[:]   = vy[:]
    diffusion.diffusion.diff_vz[:]   = vz[:]
    #
    # Install the diffusion coefficient
    #
    diffusion.diffusion.diff_dcx[:] = diffcoef[:]
    diffusion.diffusion.diff_dcy[:] = diffcoef[:]
    diffusion.diffusion.diff_dcz[:] = diffcoef[:]
    #
    # Install the source
    #
    diffusion.diffusion.diff_s[:] = source[:]
    #
    # If the initial guess (or value) for J (jinit)
    # Note: This can be either jinit (standard case) or temp
    #
    if jinit is not None:
        diffusion.diffusion.diff_j[:] = jinit[:]
    else:
        diffusion.diffusion.diff_j[:] = 0.0
    #
    # If bcoef is set, then install
    #
    if bcoef is not None:
        diffusion.diffusion.diff_b[:] = bcoef[:]
    else:
        diffusion.diffusion.diff_b[:] = 0.0
    #
    # If wcoef is set, then install
    #
    if wcoef is not None:
        diffusion.diffusion.diff_g[:] = wcoef[:]
    else:
        diffusion.diffusion.diff_g[:] = 1.0
    #
    # Now handle the boundary conditions
    #
    pp = [[diffusion.diffusion.diff_pxl, diffusion.diffusion.diff_pxr],
          [diffusion.diffusion.diff_pyl, diffusion.diffusion.diff_pyr],
          [diffusion.diffusion.diff_pzl, diffusion.diffusion.diff_pzr]]
    qq = [[diffusion.diffusion.diff_qxl, diffusion.diffusion.diff_qxr],
          [diffusion.diffusion.diff_qyl, diffusion.diffusion.diff_qyr],
          [diffusion.diffusion.diff_qzl, diffusion.diffusion.diff_qzr]]
    rr = [[diffusion.diffusion.diff_rxl, diffusion.diffusion.diff_rxr],
          [diffusion.diffusion.diff_ryl, diffusion.diffusion.diff_ryr],
          [diffusion.diffusion.diff_rzl, diffusion.diffusion.diff_rzr]]
    sgn = [-1, 1]
    for idir in range(3):
        if nn[idir] > 1:
            for ilr in range(2):
                bc = boundcond[idir][ilr]
                p  = pp[idir][ilr]
                q  = qq[idir][ilr]
                r  = rr[idir][ilr]
                if bc[0] == 'dirichlet':
                    p[:] = 0.0
                    q[:] = 1.0
                    r[:] = bc[1]
                elif bc[0] == 'dirichletrel':
                    if idir==0:
                        if ilr==0:
                            w = wcoef[0,:,:]
                        else:
                            w = wcoef[-1,:,:]
                    elif idir==1:
                        if ilr==0:
                            w = wcoef[:,0,:]
                        else:
                            w = wcoef[:,-1,:]
                    elif idir==2:
                        if ilr==0:
                            w = wcoef[:,:,0]
                        else:
                            w = wcoef[:,:,-1]
                    else:
                        stop
                    p[:] = 0.0
                    q[:] = 1.0
                    r[:] = bc[1]*w[:]
                elif bc[0] == 'flux':
                    p[:] = sgn[ilr] * bc[1]   # Because diffusion.f90 does not have the n.H, but H.
                    q[:] = bc[2]
                    r[:] = bc[3]
                else:
                    raise ValueError('Do not know boundary condition type')
    #
    # Now solve
    #
    if timedep:
        diffusion.diffusion.advdiff_solve_advectiondiffusion(convcrit, itermax, False, timestep=dt)
    else:
        diffusion.diffusion.advdiff_solve_advectiondiffusion(convcrit, itermax, False)
    #
    # If error is too large, then warn
    #
    if(diffusion.diffusion.diff_mateq_error > convcrit):
        print("Warning: Matrix equation solver did not converge. Error remains: {}".format(diffusion.diffusion.diff_mateq_error))
    # print("  Nr of iterations of matrix equation solver: {}".format(diffusion.diffusion.diff_mateq_iter))
    #
    # Copy the result
    #
    jnew = diffusion.diffusion.diff_j.copy()
    #
    # If the user wants, we compute also the F-fluxes
    #
    # Note: The flux[][:,:,:] arrays are actually the *interface*
    #       fluxes: The flux[0][0,1,1] is the flux between
    #       cell [0,1,1] and [1,1,1], and likewise for the other
    #       directions. flux[0] should actually have dimension
    #       flux[0][nx-1,ny,nz] (and likewise for other directions),
    #       but all flux[:][:,:,:] have [nx,ny,nz] dimension.
    #       Consequently flux[0][-1,:,:] = 0.0 (unused elements),
    #       and likewise flux[1][:,-1,:] = 0.0 and
    #       flux[2][:,:,-1] = 0.0.
    #
    if retflux:
        flux = [None, None, None]
        if nx > 1:
            flux[0] = diffusion.diffusion.diff_hx.copy()
        if ny > 1:
            flux[1] = diffusion.diffusion.diff_hy.copy()
        if nz > 1:
            flux[2] = diffusion.diffusion.diff_hz.copy()
    #
    # Finish the diffusion module, freeing all its memory
    #
    diffusion.diffusion.advdiff_finish()
    #
    # Return the result
    #
    # If the user wants, we return also the H-fluxes
    #
    if retflux:
        return jnew, flux
    else:
        return jnew

def solvefld3d(geom, x, y, z, alpha, source, boundcond, tinit=None, tirrad=None,
               cvrho=None, linsol_convcrit=1e-10, linsol_itermax=10000, dt=0,
               retrad=False, limiter=None, nonlin_convcrit=1e-3,
               nonlin_itermax=20, thist=False, silent=False):
    """
    A wrapper function around solvefld3d_fixedlim(), with the aim to obtain the
    full non-linear solution to the FLux-Limited-Diffusion (FLD) equation. It
    calls solvefld3d_fixedlim() each time with the new flux limiter. For more
    information, see the description of the function solvefld3d_fixedlim().
    """
    errhistory = []
    thistory   = []
    if tinit is not None:
        told    = tinit
    else:
        told    = np.zeros((len(x), len(y), len(z)))
    for nliter in range(nonlin_itermax):
        tnew, jnew, hflux, limiter = solvefld3d_fixedlim(
            geom, x, y, z, alpha, source, boundcond, tinit=told, tirrad=tirrad,
            cvrho=cvrho, convcrit=linsol_convcrit, itermax=linsol_itermax,
            dt=dt, retrad=True, limiter=limiter)
        err = (np.abs(told / tnew - 1.0)).max()
        errhistory.append(err)
        if not silent:
            print("  Nr of iter linear solver: {:4}. Flux limiter iteration: {:2}. Error = {}".format(diffusion.diffusion.diff_mateq_iter, nliter, err))
        if err < nonlin_convcrit:
            break
        told = tnew.copy()
        if thist:
            thistory.append(told)
    if thist:
        tnew = thistory
    if retrad:
        return tnew, jnew, hflux, limiter, errhistory
    else:
        return tnew


def solvefld3d_fixedlim(geom, x, y, z, alpha, source, boundcond, tinit=None,
                        tirrad=None, cvrho=None, convcrit=1e-10, itermax=10000,
                        dt=0, retrad=False, limiter=None):
    """
    Solve the Flux-Limited Diffusion (FLD) equation for radiative transfer. This is a
    version of the diffusion equation, but specifically for radiative transfer. This
    is the linear version, where the flux limited is a given quantity and kept fixed.
    The new value of the flux limiter is computed at the end of this subroutine.
    Iteration of this procedure, with new value of the limiter at each step, leads
    to the full FLD solution.

    NOTE: CGS units used here!

    We follow the paper by Kuiper, Klahr, Dullemond, Kley & Henning (2010).

    Arguments:
    ----------

    geom : str
        'cartesian'   = Cartesian coordinates
        'spherical'   = Spherical coordinates with x=r, y=theta and z=phi
        'cylindrical' = Cylindrical coordinates with x=r, y=phi and z=z

    x : array
        Array of values of x. If just one value: dimension is not active.

    y : array
        Array of values of y. If just one value: dimension is not active.

    z : array
        Array of values of z. If just one value: dimension is not active.

    alpha : array
        Array of values of the rho*kappa opacity.

    source : array
        Array of values of the source term S(x,y,z)

    boundcond : list of lists
        [bcx,bcy,bcz]. Example bcx (boundary conditions in x direction):
        bcx is a list of boundary conditions, one for left, one for right
        bcx = [bcxl,bcxr]. Examples for left:
        bcxl = ['dirichlet',value] : The left x boundary is a Dirichlet
                                     boundary condition with J=value. The
                                     value can be either a number, or an array
                                     of numbers (in this case an ny,nz array).
        bcxl = ['flux',p,q,r] :      More general bc: p n.Flux + q J  = r, where
                                     Flux = D Grad J, and n is the outward-pointing
                                     normal vector.

    tinit : array
        Array of initial (guess of) the temperature

    tirrad : array
        Array of irradiation temperature (the optically thin region of the cloud)

    cvrho : array
        Array of values of the c_v * rho(x,y,z) (only for time-dependent) (If None, then taken to be 1)

    convcrit : float
        Convergence criterion to be used for the linear equation solver.

    itermax : int
        Maximum number of convergence steps

    dt : float
        Time step for time-dependent case. If dt=0, then stationary case.

    retrad : bool
        If True, then not only T but also J, Hx, Hy, Hz are returned.

    limiter : float

        The value of the flux limiter, defined as 3*lambda, where lambda is from
        Levermore & Pomraning (1981) ApJ 248, 321, their Eq.(28), with R defined
        in Eq.(18). The simplest way to compute this is to set retrad=True, which
        not only returns the new temperature, but also J, H and the limiter
        according to Levermore & Pomraning Eq.(28) to be used for the next call.
        Note that in highly optically thick regions, limiter --> 1.0, while in
        very optically thin regions, limiter --> 0.0.
    """
    #
    # Some constants
    #
    tmin     = 1e-2                 # Some lower bound to the temperature
    pisigsb  = 55404.34639419066e0  # = pi/sigma_SB in CGS units
    cc       = 2.9979245800000e10   # Light speed             [cm/s]
    aa       = 7.5657e-15           # 4 ss / cc               [erg/cm^3/K^4]
    # ss       = 5.6703e-5            # Stefan-Boltzmann const  [erg/cm^2/K^4/s]
    #
    # Find length of all dimensions
    #
    nx = len(x)
    ny = len(y)
    nz = len(z)
    assert nx > 0, 'Each dimension must have at least one value of its coordinate'
    assert ny > 0, 'Each dimension must have at least one value of its coordinate'
    assert nz > 0, 'Each dimension must have at least one value of its coordinate'
    # nn = [nx, ny, nz]
    #
    # Interpret time dependency
    #
    if dt != 0:
        assert tinit is not None, "For time-dependent flux-limited diffusion we need an initial temperature"
    #
    # If initial temperature tinit is given, then compute the corresponding mean intensity J
    #
    if tirrad is None:
        tirrad = 0.0
    if tinit is not None:
        jinit = (tinit**4 - (tirrad + tmin)**4) / pisigsb
    else:
        jinit = None
    #
    # Compute the g factor
    #
    #   g = 1 / ( c * f_c )  with f_c defined below Eq. 6 of Kuiper etal
    #
    if cvrho is not None and tinit is not None:
        gcoef = (1.0 + cvrho / (4 * aa * tinit**3)) / cc
    else:
        gcoef = 1.0
    #
    # Now compute the flux-limited diffusion coefficient
    #
    if limiter is None:
        limiter = 1.0                  # The optically thick value
    diffcoef = limiter / alpha / 3.0   # Optically thick means: D = (1/3)/alpha
    #
    # Now call the above diffusion solver
    #
    jnew, hflux = solvediff3d(geom, x, y, z, diffcoef, source, boundcond,
                              jinit=jinit, gcoef=gcoef, convcrit=convcrit,
                              itermax=itermax, dt=dt, retflux=True)
    #
    # Reconstruct the temperature again
    #
    tnew = (jnew * pisigsb + (tirrad + tmin)**4)**0.25
    #
    # If the user wants to have an estimate of the flux limiter for the next
    # call, compute it here
    #
    if retrad:
        #
        # Compute the flux limiter
        #
        hav  = np.zeros_like(diffcoef)
        ndim = 0
        if nx > 1:
            hav[1:-1, :, :] += 0.25 * (hflux[0][1:nx - 1, :, :] + hflux[0][0:nx - 2, :, :])**2
            ndim += 1
        if ny > 1:
            hav[:, 1:-1, :] += 0.25 * (hflux[1][:, 1:ny - 1, :] + hflux[1][:, 0:ny - 2, :])**2
            ndim += 1
        if nz > 1:
            hav[:, :, 1:-1] += 0.25 * (hflux[2][:, :, 1:nz - 1] + hflux[2][:, :, 0:nz - 2])**2
            ndim += 1
        hav        = hav**0.5
        gradjdiva  = 3 * hav / limiter                            # Equivalent to | grad J / alpha |
        rlim       = gradjdiva / jnew                             # Levermore & Pomraning Eq.(18)
        limiternew = (2.0 + rlim) / (6.0 + 3.0 * rlim + rlim**2)  # Levermore & Pomraning Eq.(28)
        limiternew *= 3       # Because the factor 1/3 is already accounted for in the diffusion module: limiter=3*lambda.
        #
        # Copy the limiternew to the boundary ghost cells
        #
        if nx > 1:
            limiternew[0, :, :]  = limiternew[1, :, :]
            limiternew[-1, :, :] = limiternew[-2, :, :]
        if ny > 1:
            limiternew[:, 0, :]  = limiternew[:, 1, :]
            limiternew[:, -1, :] = limiternew[:, -2, :]
        if nz > 1:
            limiternew[:, :, 0]  = limiternew[:, :, 1]
            limiternew[:, :, -1] = limiternew[:, :, -2]
    #
    # Return the result
    #
    # If the user wants, we return also the J and the H-fluxes, as well as the limiter
    #
    if retrad:
        return tnew, jnew, hflux, limiternew
    else:
        return tnew
