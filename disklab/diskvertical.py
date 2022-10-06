import numpy as np
from . import natconst as nc
from .grainmodel import GrainModel
from .solvediffonedee import solvediffonedee
from .meanopacity import evaluate_meanopacity


class DiskVerticalModel(object):
    """
    This is a 1-D vertical model of a protoplanetary disk, i.e. at a given radius in the disk.
    The vertical coordinate is treated in the z<<r limit, i.e. no geometric terms due to
    spherical vs cylindrical coordinate are taken into account.

    Arguments:
    ----------

    mstar : float
        Stellar mass [g]

    r : float
        Radius at which the vertical model is located [cm]

    siggas : float
        Surface density of the gas [g/cm^2]


    Keywords:
    ---------

    nz : int
        Nr of vertical grid points from the midplane to the top

    zrmaz : float
        Upper z of the grid in units of r, i.e. maximum z/r of the grid

    zgrid : array
        Alternative to nz and zrmax you can also give the z grid yourself

    lstar : float
        Stellar luminosity [erg/s]

    flidx : float
        Flaring irradiation index such that flaring angle is flang = flidx * z / r

    meanopacitymodel : list
        the mean opacity parameters, see `meanopacity.evaluate_meanopacity`

    mugas : float
        the mean molecular weight in units of mp

    gamma : float
        The ratio of specific heats

    alphavisc : float
        The viscous alpha parameter

    init_opacity : bool
        by default, if meanopacity parameters are set, the mean opacity is
        evaluated. In some cases, other things need to happen before the
        opacity can be calculated, for example dust species might need to be
        added first. In that case, the call to mean opacity fails. To prevent
        this, you can turn it off by setting `init_opacity=False`

    """

    def __init__(self, mstar, r, siggas, nz=90, zrmax=0.4, zgrid=None,
                 lstar=nc.LS, flidx=None, flang=None, meanopacitymodel=None,
                 mugas=2.3, gamma=(7. / 5.), alphavisc=None, Sc=1.0, tgas=None,
                 init_opacity=True):
        #
        # Create the grid
        #
        assert np.isscalar(r), "Error: DiskVerticalModel only works for a single radius."
        self.r = r    # Warning: do not change this after __init__. Better make new DiskVerticalModel.
        if zgrid is None:
            self.z = np.linspace(0., zrmax * r, nz)
        else:
            self.z = zgrid.copy()
        zmax       = self.z.max()
        zi         = 0.5 * (self.z[1:] + self.z[:-1])
        zi         = np.hstack((0., zi, zmax))
        self.dz    = zi[1:] - zi[:-1]
        #
        # Copy the relevant quantities into the object
        #
        self.mstar = mstar
        self.siggas = siggas
        self.lstar = lstar
        if flidx is not None and flang is not None:
            raise ValueError("Cannot specify both flaring index and flaring angle.")
        if flidx is None and flang is None:
            self.flidx = 0.1
        else:
            if flidx is not None:
                self.flidx = flidx
            if flang is not None:
                self.flang = flang
        if alphavisc is not None:
            self.alphavisc = alphavisc * np.ones_like(self.z)
        #
        # Distinguish between cylindrical radius r and spherical radius r_spher
        #
        self.r_spher = np.sqrt(self.r**2 + self.z**2)
        #
        # Some defaults
        #
        self.mugas = mugas
        self.gamma = gamma
        self.Sc = Sc
        #
        # Some computations
        #
        self.omk_midpl = np.sqrt(nc.GG * self.mstar / self.r**3)
        self.omk_full  = np.sqrt(nc.GG * self.mstar / self.r_spher**3)
        #
        # First guess of structure
        #
        if tgas is not None:
            if flang is not None:
                raise ValueError('both tgas and flang are given')
        else:
            if flang is None:
                flang = 0.05
            tgas = (0.5 * flang * self.lstar / (nc.ss * 4 * nc.pi * self.r**2))**0.25

        self.compute_rhogas_isothermal(tgas)
        #
        # Opacity model
        #
        if meanopacitymodel is not None:
            self.meanopacitymodel = meanopacitymodel
            if init_opacity:
                try:
                    self.compute_mean_opacity()
                except KeyError:
                    raise
                except Exception as error:
                    print('mean opacity failed - you might want to set `init_opacity=False`')
                    print('opacity failed with following error')
                    print(error)

    def vertically_integrate(self, rho):
        """
        Integrate some density-like quantity rho vertically, to obtain a surface density-like
        quantity. We of course multiply by 2 due to the upper/lower part of the disk.
        """
        q           = rho * self.dz
        return 2 * q.sum()

    def compute_surfacedensity(self):
        """
        Integrates the density vertically (multiplied by 2, because of the two sides
        of the disk) to obtain the surface density.
        """
        self.sigmagas = 2*(self.dz*self.rhogas).sum()
    
    def compute_rhogas_isothermal(self, tgas, include_curvature=True):
        """
        Set up the gas density according to a vertically isothermal hydrostatic equilibrium.

        ARGUMENTS:
           tgas                     Scalar giving the temperature [K] of the gas.

        OPTIONAL ARGUMENTS:
           include_curvature        If False: Use midplane value of Omega_K=G*M/r**3, and do
                                              ignore difference between spherical and cylindrical
                                              radius. The vertical gravitational force then will
                                              go exactly proportional to -z (perfect harmonic
                                              oscillator potential). For geometrically thin disks
                                              this is reasonably accurate.
                                    If True:  Include the exact vertical dependence of the
                                              gravitational force according to the formula
                                              -(G*M/r_spher**3)*z.
        """
        assert np.isscalar(tgas), "For isothermal model, tgas must be vertically constant"
        self.tgas   = np.ones_like(self.z) * tgas
        if not include_curvature:
            #
            # Without the curvature the solution is a simple Gaussian
            #
            cs          = np.sqrt(nc.kk * tgas / (self.mugas * nc.mp))
            hp          = cs / self.omk_midpl
            self.rhogas = np.exp(-0.5 * self.z**2 / hp**2)
            dum         = self.vertically_integrate(self.rhogas)
            self.rhogas = self.rhogas * self.siggas / dum + nc.flr
        else:
            #
            # With curvature this is more complicated, so simply call the full hydrostatic method
            #
            self.compute_rhogas_hydrostatic(include_curvature=True)

    def compute_rhogas_hydrostatic(self, include_curvature=True, add_selfgrav=False,
                                   dtg=None):
        """
        Compute the rhogas according to hydrostatic equilibrium, for a given temperature
        structure. Before calling, you must have specified self.tgas: an array of gas
        temperature as a function of z.

        OPTIONAL ARGUMENTS:
           include_curvature        If False: Use midplane value of Omega_K=G*M/r**3, and do
                                              ignore difference between spherical and cylindrical
                                              radius. The vertical gravitational force then will
                                              go exactly proportional to -z (perfect harmonic
                                              oscillator potential). For geometrically thin disks
                                              this is reasonably accurate.
                                    If True:  Include the exact vertical dependence of the
                                              gravitational force according to the formula
                                              -(G*M/r_spher**3)*z.
           add_selfgrav             If True:  Add the self.selfgrav_fz force to the vertical
                                              force. IMPORTANT: It does not compute self.selfgrav_fz.
                                              The computation has to be done in advance with the 
                                              method self.compute_selfgravity_vertical_force_1d()
                                              or with the fully 2-D methods from disk2d.py.
           dtg                      If set:   Include the weight of the dust in the hydrostatic
                                              equilibrium. Note that if dtg is an array, the dtg
                                              is given on the z-grid, and is thus not "attached"
                                              to the gas column density. See the subroutine
                                              compute_rhogas_hydrostatic_with_dust_and_tracers()
                                              for a way to handle this in an "attached" way, or
                                              in other words, in a "comoving" way.
        """
        assert hasattr(self, 'tgas'), "Error: Can only compute hydrostatic equilibrium if temperature tgas is given as an array."
        if np.isscalar(self.tgas):
            self.tgas = np.ones_like(self.z) * self.tgas
        if not hasattr(self, 'rhogas'):
            self.rhogas = np.zeros_like(self.z)
        if not include_curvature:
            omk2    = self.omk_midpl**2 + np.zeros_like(self.z)
        else:
            omk2    = self.omk_full**2
        z           = self.z
        dzi         = z[1:] - z[:-1]
        grav_fz     = - z * omk2 * self.mugas * nc.mp / nc.kk
        if add_selfgrav:
            grav_fz += self.selfgrav_fz * self.mugas * nc.mp / nc.kk
        if dtg is not None:
            grav_fz *= (1.0+dtg)
        tgas        = self.tgas
        rho         = self.rhogas
        if rho[0]==0: rho[0] = 1.0
        for iz in range(len(self.z) - 1):
            dlgt        = (np.log(tgas[iz + 1]) - np.log(tgas[iz])) / dzi[iz]
            grv         = - 0.5 * ( grav_fz[iz + 1] / tgas[iz + 1] + grav_fz[iz] / tgas[iz] )
            rho[iz + 1] = rho[iz] * np.exp(-dzi[iz] * (grv + dlgt))
        dum             = self.vertically_integrate(self.rhogas)
        self.rhogas     = self.rhogas * self.siggas / dum + nc.flr

    def compute_rhogas_hydrostatic_with_dust_and_tracers(self, include_curvature=True,
                                                         add_selfgrav=False, dtgitermax=20,
                                                         dtgerrtol=1e-4,incl_dustweight=True,
                                                         comove_dust=True,comove_tracers=True):
        """
        As compute_rhogas_hydrostatic(), but now including the dust and (possibly)
        trace species. The dust and the trace species will be moved along with the
        gas, as the gas readjusts to a new hydrostatic equilibrium.

        Note that the comoving of the dust and the tracers is not done in a strictly
        mass-conserving way!

        ARGUMENTS:

          incl_dustweight:   If True, then include the dust weight in the hydrostatic equilibrium.
          comove_dust:       If True, then move the dust along with the gas as the latter finds a new
                             hydrostatic equilibrium.
          dtgitermax:        The maximum nr of iterations on the dust weight contribution.
          dtgerrtol:         The error tolerance on the dust weight contribution.
        """
        assert self.rhogas.max()>0, 'Cannot create cumulative density if rhogas not yet specified.'
        sig_old = self.get_cumuldens()
        if incl_dustweight and hasattr(self,'dust'):
            assert hasattr(self,'rhogas'), 'Cannot do hydrostatics with dust if no initial vertical structure is set.'
            if hasattr(self,'sigmagas'):
                sigmagas = 2*(self.dz*self.rhogas).sum()
                assert np.abs(sigmagas/(self.sigmagas+nc.flr)-1.)<1e-4, 'Error: Vertical gas density distribution incompatible with sigmagas.'
            rhodust = np.zeros_like(self.rhogas)
            rhog_old = self.rhogas.copy()
            for d in self.dust:
                rhodust += d.rho
            dtg_old    = rhodust / self.rhogas
            dtg        = dtg_old.copy()
            for iter in range(dtgitermax):
                dtg_prev = dtg.copy()
                self.compute_rhogas_hydrostatic(include_curvature=include_curvature,
                                                add_selfgrav=add_selfgrav,dtg=dtg)
                sig = self.get_cumuldens()
                dtg = np.interp(sig,sig_old,dtg_old)
                err = np.abs(dtg-dtg_prev).max()
                if err<dtgerrtol: break
            else:
                raise ValueError('No convergence in compute_rhogas_hydrostatic_with_dust_and_tracers(): Error = {}'.format(err))
        else:
            self.compute_rhogas_hydrostatic(include_curvature=include_curvature,
                                            add_selfgrav=add_selfgrav)
            sig = self.get_cumuldens()
        if comove_dust and hasattr(self,'dust'):
            for d in self.dust:
                ddtg_old = d.rho/rhog_old
                ddtg_new = np.interp(sig,sig_old,ddtg_old)
                d.rho    = self.rhogas*ddtg_new
        # Next do the tracers here (todo in future; template already given here)
        # if comove_tracers and hasattr(self,'tracers'):
        #     for t in self.tracers:
        #         tttg_old = t.rho/rhog_old
        #         tttg_new = np.interp(sig,sig_old,tttg_old)
        #         t.rho    = self.rhogas*tttg_new

    def compute_rhogas_hydrostatic_adiabatic(self, tol=1e-10, maxiter=100, include_curvature=True,
                                             add_selfgrav=False):
        """
        As compute_rhogas_hydrostatic(), but now we first compute the specific
        entropy at each point, then put that on a mass-coordinate (i.e. fixed
        to the gas packet), then iterate. Essentially this readjusts the density
        to a hydrostatic equilibrium but it does so in an adiabatic manner.

        ARGUMENTS:
           tol                      The tolerance level at which to finish the iteration.
           maxiter                  The maximum nr of iterations. If exceeded, an error is thrown.

        OPTIONAL ARGUMENTS:
           include_curvature        If False: Use midplane value of Omega_K=G*M/r**3, and do
                                              ignore difference between spherical and cylindrical
                                              radius. The vertical gravitational force then will
                                              go exactly proportional to -z (perfect harmonic
                                              oscillator potential). For geometrically thin disks
                                              this is reasonably accurate.
                                    If True:  Include the exact vertical dependence of the
                                              gravitational force according to the formula
                                              -(G*M/r_spher**3)*z.
           add_selfgrav             If True:  Add the self.selfgrav_fz force to the vertical
                                              force. IMPORTANT: It does not compute self.selfgrav_fz.
                                              The computation has to be done in advance with the 
                                              method self.compute_selfgravity_vertical_force_1d()
                                              or with the fully 2-D methods from disk2d.py.
        """
        assert hasattr(self, 'tgas'), "Error: Can only compute hydrostatic equilibrium if temperature tgas is given as an array."
        if np.isscalar(self.tgas):
            self.tgas = np.ones_like(self.z) * self.tgas
        if not hasattr(self, 'rhogas'):
            self.rhogas = np.zeros_like(self.z)
        #
        # Get the grid
        #
        z           = self.z
        nz          = len(z)
        dz          = self.dz
        # zi          = 0.5 * (z[1:] + z[:-1])
        dzi         = z[1:] - z[:-1]
        #
        # Compute the specific entropy everywhere
        #
        self.compute_specific_entropy_from_tgas()
        entropy     = self.specific_entropy
        entropyi    = np.hstack((entropy[0], 0.5 * (entropy[1:] + entropy[:-1]), entropy[nz - 1]))
        #
        # Compute the column density coordinate
        #
        nz          = len(self.z)
        coldensi    = np.zeros(nz + 1)
        for iz in range(nz - 1, -1, -1):
            coldensi[iz] = coldensi[iz + 1] + self.dz[iz] * self.rhogas[iz]
        if np.abs(2 * coldensi[0] / self.siggas - 1.0) > 1e-3:
            raise ValueError('Error: Column density of rhogas unequal to siggas.')
        #
        # Now compute the new hydrostatic balance
        #
        if not include_curvature:
            omk2    = self.omk_midpl**2 + np.zeros_like(self.z)
        else:
            omk2    = self.omk_full**2
        grav_fz     = - z * omk2 * self.mugas * nc.mp / nc.kk
        if add_selfgrav:
            grav_fz += self.selfgrav_fz * self.mugas * nc.mp / nc.kk
        tgas        = self.tgas
        rho         = self.rhogas
        tprev       = self.tgas.copy()
        for iter in range(maxiter):
            #
            # Do a hydrostatic equilibrium integration
            #
            rho[0]      = 1.0
            for iz in range(len(self.z) - 1):
                dlgt        = (np.log(tgas[iz + 1]) - np.log(tgas[iz])) / dzi[iz]
                grv         = - 0.5 * ( grav_fz[iz + 1] / tgas[iz + 1] + grav_fz[iz] / tgas[iz] )
                rho[iz + 1] = rho[iz] * np.exp(-dzi[iz] * (grv + dlgt))
                dum         = self.vertically_integrate(self.rhogas)
                rho[:]      = rho[:] * self.siggas / dum + nc.flr
            #
            # Compute the new column density
            #
            coldensnew     = np.zeros(nz)
            coldensnew[-1] = 1e-90
            for iz in range(nz - 2, -1, -1):
                coldensnew[iz] = coldensnew[iz + 1] + 0.5 * (dz[iz + 1] * rho[iz + 1] + dz[iz] * rho[iz])
            coldensnew[0] = (1.0 - 1e-14) * coldensi[0]
            #
            # Map the specific entropy back to the z-grid
            #
            entropy[:] = np.interp(-coldensnew, -coldensi, entropyi)
            #
            # Compute the new temperature
            #
            self.compute_tgas_from_specific_entropy()
            #
            # Check error
            #
            error = np.abs((self.tgas - tprev) / (self.tgas + tprev)).max()
            if error < tol:
                break
            tprev = self.tgas.copy()
        self.iter = iter

    def compute_selfgravity_vertical_force_1d(self,incl_gas=True,incl_dust=True,cellwalls=False):
        """
        In principle self-gravity is a global force, and in disk2d.py there are various
        methods to compute this. But a simple version of the vertical force through
        self-gravity can be computed in the infinite plane-parallel slab approximation.
        This is what is done here.
        """
        self.rho = np.zeros_like(self.rhogas)
        if incl_gas: self.rho += self.rhogas
        if incl_dust and hasattr(self, 'dust'):
            if type(self.dust) == list:
                for d in self.dust:
                    self.rho += d.rho
            else:
                self.rho += self.dust.rho
        self.selfgrav_fz = np.zeros(len(self.z)+1)
        for iz in range(1,len(self.z)+1):
            self.selfgrav_fz[iz] = self.selfgrav_fz[iz-1] + 2*self.dz[iz-1]*self.rho[iz-1]
        self.selfgrav_fz *= -2*np.pi*nc.GG
        if not cellwalls:
            self.selfgrav_fz = 0.5 * ( self.selfgrav_fz[1:] + self.selfgrav_fz[:-1] )

    def compute_specific_entropy_from_tgas(self):
        """
        Thermodynamic function computing the specific entropy s from the gas temperature tgas.
        """
        if not hasattr(self, 'specific_entropy'):
            self.specific_entropy = np.zeros_like(self.z)
        rhogas = self.rhogas
        gamma  = self.gamma
        mugas  = self.mugas
        tgas   = self.tgas
        s      = self.specific_entropy
        pgas   = rhogas * tgas * nc.kk / (mugas * nc.mp)
        K      = pgas / rhogas**gamma
        s[:]   = (nc.kk / (mugas * (gamma - 1))) * np.log(K)

    def compute_tgas_from_specific_entropy(self):
        """
        Thermodynamic function computing the gas temperature from the specific entropy s
        """
        rhogas  = self.rhogas
        gamma   = self.gamma
        mugas   = self.mugas
        s       = self.specific_entropy
        tgas    = self.tgas
        K       = np.exp(((mugas * (gamma - 1)) / nc.kk) * s)
        pgas    = K * rhogas**gamma
        tgas[:] = (pgas / rhogas) * (mugas * nc.mp / nc.kk)

    def compute_mean_opacity(self, meanopacitymodel=None):
        """
        To compute the radiative heating and cooling of a protoplanetary disk, we need
        the opacity of the disk material. This is most likely the dust, but it could also
        be gas opacity for the hot regions. Or you could decide to specify a simple opacity
        law independent of the dust species in the disk.

        The method compute_mean_opacity() creates and fills two arrays:

           self.mean_opacity_planck[0:nz]
           self.mean_opacity_rosseland[0:nz]

        which are the Planck-mean and Rosseland-mean opacities respectively. They are
        normalized to the *gas* density, i.e. the Rosseland extinction coefficient would be:

           extinction_coefficient[0:nz] = self.rhogas[0:nz] * self.mean_opacity_rosseland[0:nz]

        and the total vertical Rosseland optical depth would then be

           tau_rosseland_total = 2 * ( extinction_coefficient * self.dz ).sum()

        where the factor of 2 is simply because we only model the upper half of the disk.

        The normalization to the gas density (instead of the dust density) is simply
        because "the" gas of the disk is just a single entity, whereas DISKLAB allows
        multiple dust species. The mean opacity is thus "cross section per gram of gas",
        but it will (or may) include any opacity bearing material suspended in the gas.
        So for a typical dust opacity we get that the mean opacity is the dust-to-gas
        ratio times the dust opacity.

        Note that if the temperature of the disk changes, you will have to call this
        method compute_mean_opacity() again to recompute the opacity arrays. This will
        typically happen when doing a vertical structure iteration, where the temperature
        changes with iteration step.

        Note also that if the opacities depend on a z-dependent distribution of disk
        components (e.g. on dust settling or on a z-dependent disk chemical composition),
        then one also has to re-call this method every time this vertical distribution
        changes.

        Calling compute_mean_opacity() will automatically delete the previously installed
        mean opacities, so you can call compute_mean_opacity() as often as you like,
        without worrying about cluttering memory.

        Tip: Better call the method compute_mean_opacity() too often than too few times,
             because if you forget to call compute_mean_opacity() after a change in the
             disk structure, you may get wrong results because the mean opacities will
             be wrongly distributed vertically.

        The method compute_mean_opacity() is a very general method: it will allow
        different kinds of opacity models to be installed.

        ARGUMENTS:

          meanopacitymodel  A list. First element is a string containing the name of the
                            opacity model. The further elements contain information used
                            by that opacity model, and these elements can be different
                            for different opacity models. They are described below.

                            ---> If meanopacitymodel is None, then this method will use
                            self.meanopacitymodel

        OPACITY MODELS AVAILABLE:

          'supersimple'     The simplest opacity model available: you simply specify
                            a value to be used. The second element of the meanopacitymodel
                            list is a dictionary with information about this opacity.
                            The simplest way is:

                             meanopacitymodel = ['supersimple',{'kappagas':1e0}]

                            You can also specify a hypothetical dust opacity, and a dust-to-gas
                            ratio:

                             meanopacitymodel = ['supersimple',{'dusttogas':0.01,'kappadust':1e2}]

                            (note that both examples give the same result).

                            You can also use this model to specify a z-dependent opacity,
                            simply by giving an array instead of a value:

                             kappaarray       = np.ones_like(self.rhogas)
                             meanopacitymodel = ['supersimple',{'kappagas':kappaarray}]

                            where of course you set kappaarray to some more useful
                            array than just np.ones_like(self.rhogas), as long as it
                            has the same number of elements as self.rhogas.

          'dustcomponents'  This will construct the mean opacity arrays from the available
                            dust components in the disk (or from the dust components you
                            explicitly give as arguments to this method). The second
                            element of the list is a dictionary with further settings.
                            The most important dictionary element is 'method'. It can
                            have the following values:

                            method = 'fullcalculation':
                                      Compute the mean opacities from the full
                                      frequency-dependent opacities of the dust components.
                                      This model is very time-consuming, because it will
                                      recalculate the integrals over the full frequency grids.
                                      For fast calculations of disk models, this is not ideal.
                                      But there can be circumstances by which this is the
                                      better method: if different dust species mix/drift
                                      in different ways, so that the composition of the
                                      dust mixture is different at different times.

                            method = 'simplemixing':
                                      Use mean opacities from each of the dust species
                                      individually, and average them weighted by abundance.
                                      For a single dust species this is of course exact.
                                      For multiple dust species this is not quite correct,
                                      but it is fast. The reason why it is not quite
                                      correct is that the Rosseland mean of the mixed
                                      frequency dependent opacity is not equal to the
                                      average of the Rosseland mean of the individual
                                      dust opacities. It is only correct if the frequency-
                                      dependent opacities are 'correlated'. In atmospheric
                                      radiative transfer physics they call this the
                                      'correlated k assumption'. It is the method of
                                      choice if one wants to allow time- and space-
                                      varying dust abundances, while avoiding the time-
                                      consuming 'fullcalculation' method.

                            So if you choose 'simplemixing' then this if what the meanopacitymodel
                            should look like:

                             meanopacitymodel = ['dustcomponents',{'method':'simplemixing'}]

          'tabulated'       This method will look up the opacity from a table that you
                            provide, and it will interpolate in that table. The meanopacitymodel
                            has to contain this table:

                             meanopacitymodel = ['tabulated',{'rhogrid':rhogas,'tempgrid':temp, \
                                                              'kappa_planck':kappa_planck,      \
                                                              'kappa_rosseland':kappa_rosseland,\
                                                              'method':'linear'}]

                            where rhogas[0:nrho] and temp[0:ntemp] are two 1-D arrays giving the
                            coordinates of the table, and kappa_planck[0:nrho,0:ntemp] and
                            kappa_rosseland[0:nrho,0:ntemp] are the 2-D tables of opacity in
                            units of cm^2/gram-of-gas.

                            Note that since these opacities can be extremely steep functions of
                            density and temperature, it is often better to do the tabulation in
                            logarithmic form. You can do this, by specifying in the meanopacitymodel
                            instead of 'kappa_planck' and 'kappa_rosseland' the arrays
                            'ln_kappa_planck' and 'ln_kappa_rosseland', which then should be the
                            natural logarithms of kappa_planck and kappa_rosseland, respectively:

                             ln_kappa_planck    = np.log(kappa_planck)
                             ln_kappa_rosseland = np.log(kappa_rosseland)
                             meanopacitymodel = ['tabulated',{'rhogrid':rhogas,'tempgrid':temp,       \
                                                              'ln_kappa_planck':ln_kappa_planck,      \
                                                              'ln_kappa_rosseland':ln_kappa_rosseland,\
                                                              'method':'linear'}]

                            The resulting interpolated values will then automatically be
                            np.exp(...) again, so that the result is again the actual opacity instead
                            of the np.log(...) of the opacity.

          'belllin'         This is the well-known Bell & Lin opacity table.
                            Set meanopacitymodel to:

                             meanopacitymodel = ['belllin']

        """
        #
        # Make sure that the mean opacity arrays exist
        #
        if not hasattr(self, 'mean_opacity_planck'):
            self.mean_opacity_planck = np.zeros_like(self.rhogas)
        if not hasattr(self, 'mean_opacity_rosseland'):
            self.mean_opacity_rosseland = np.zeros_like(self.rhogas)
        #
        # Get meanopacitymodel
        #
        if meanopacitymodel is None:
            assert hasattr(self, 'meanopacitymodel'), 'To install the mean opacities you must given an opacity model.'
            meanopacitymodel = self.meanopacitymodel
        #
        # Check if meanopacitymodel is a scalar
        #
        if np.isscalar(meanopacitymodel):
            meanopacitymodel = [meanopacitymodel]
        self.meanopacitymodel = meanopacitymodel
        #
        # If we have dust components, then extract information from them
        #
        rhodust = None
        grain   = None
        if hasattr(self, 'dust'):
            rhodust = []
            grain   = []
            if type(self.dust) == list:
                for d in self.dust:
                    if hasattr(d, 'rho'):
                        rhodust.append(d.rho)
                    else:
                        rhodust.append(None)
                    if hasattr(d, 'grain'):
                        grain.append(d.grain)
                    else:
                        grain.append(None)
        #
        # Check if we have the midplane gas density and temperature available
        #
        if hasattr(self, 'rhogas'):
            rhogas = self.rhogas
        else:
            rhogas = None
        if hasattr(self, 'tgas'):
            tgas = self.tgas
        else:
            tgas = None
        #
        # Now handle the different meanopacitymodels:
        #
        meanopac = evaluate_meanopacity(meanopacitymodel, rhogas, tgas, rhodust=rhodust, grain=grain)
        self.mean_opacity_planck[:]    = meanopac['planck']
        self.mean_opacity_rosseland[:] = meanopac['rosseland']

    def irradiate_with_flaring_index(self):
        """
        Compute the absorbed stellar radiation in the surface layer. The albedo is assumed to be zero, so
        that all the stellar radiation is absorbed. The irradiation is done in a flaring-index kind of
        way: irradiation from the top with an irradiation angle.
        """
        z            = self.z
        nz           = len(z)
        if not hasattr(self, 'irrad_src'):
            self.irrad_src     = np.zeros(nz)
        if not hasattr(self, 'irrad_jmean'):
            self.irrad_jmean   = np.zeros(nz)
        if not hasattr(self, 'irrad_taugraz'):
            self.irrad_taugraz = np.zeros(nz + 1)
        if not hasattr(self, 'irrad_tauvert'):
            self.irrad_tauvert = np.zeros(nz + 1)
        if not hasattr(self, 'irrad_flux'):
            self.irrad_flux    = np.zeros(nz + 1)
        if not hasattr(self, 'irrad_flang'):
            self.irrad_flang   = np.zeros(nz)
        r            = self.r
        dz           = self.dz
        rhokap       = self.rhogas * self.mean_opacity_planck
        flux         = self.lstar / (4 * nc.pi * r**2)
        src          = self.irrad_src
        jmean        = self.irrad_jmean
        self.irrad_flux[nz] = flux
        for iz in range(nz - 1, -1, -1):
            if hasattr(self, 'flidx'):
                flang    = self.flidx * z[iz] / r
            else:
                flang    = self.flang
            dtau         = rhokap[iz] * dz[iz] / (flang + 1e-10)
            fluxprev     = flux
            flux        *= np.exp(-dtau)
            src[iz]      = (fluxprev - flux) * flang / dz[iz]
            jmean[iz]    = 0.5 * (fluxprev + flux) / (4 * nc.pi)
            self.irrad_flux[iz]    = flux
            self.irrad_taugraz[iz] = self.irrad_taugraz[iz + 1] + dtau
            self.irrad_tauvert[iz] = self.irrad_tauvert[iz + 1] + dtau * (flang + 1e-10)
            self.irrad_flang[iz]   = flang

    def compute_local_shear_viscosity(self):
        """
        Given the Shakura-Sunyaev alpha parameter self.alphavisc, compute and store
        the viscosity self.nuvisc = alpha*cs^2/omk
        """
        z                = self.z
        nz               = len(z)
        assert hasattr(self, 'alphavisc'), "Error: Cannot compute viscosity if alphavisc is not set."
        if not hasattr(self, 'nuvisc'):
            self.nuvisc = np.zeros(nz)
        alpha            = self.alphavisc
        tgas             = self.tgas
        cs2              = nc.kk * tgas / (self.mugas * nc.mp)
        omk              = self.omk_midpl
        self.nuvisc[:]   = alpha * cs2 / omk

    def compute_viscous_heating(self):
        """
        Compute the viscous heating source term. This is implemented here is a
        simple way: q = (9/4)*rhogas*nu*omk^2 with nu=nuvisc computed by
        the compute_local_shear_viscosity() routine.
        """
        z                = self.z
        nz               = len(z)
        if not hasattr(self, 'visc_src'):
            self.visc_src = np.zeros(nz)
        self.compute_local_shear_viscosity()
        omk              = self.omk_midpl
        rho              = self.rhogas
        nu               = self.nuvisc
        self.visc_src[:] = (9. / 4.) * rho * nu * omk**2

    def solve_vert_rad_diffusion(self):
        """
        Given a possible irrad_src and a possible visc_src, integrate the diffusion equation to
        get the diffuse radiation field. This is the stationary state case. For a time-dependent
        time step, use timestep_vert_rad_diffusion()
        """
        if not hasattr(self, 'diff_jmean'):
            self.diff_jmean  = np.zeros_like(self.z)
        jmean        = self.diff_jmean
        z            = self.z
        nz           = len(z)
        #
        # The opacity at the interfaces (note: nz-1 elements)
        #
        rhokappa     = self.rhogas * self.mean_opacity_rosseland
        meanalphai   = 0.5 * (rhokappa[1:] + rhokappa[:-1])
        #
        # Compute or get the source
        #
        src          = np.zeros_like(self.z)
        if hasattr(self, 'irrad_src'):
            src += self.irrad_src
        if hasattr(self, 'visc_src'):
            src += self.visc_src
        #
        # Integrate the flux at the interfaces (note: here nz+1 elements)
        #
        if not hasattr(self, 'diff_hflux'):
            self.diff_hflux    = np.zeros(nz + 1)
        hflux = self.diff_hflux
        dz    = self.dz
        hflux[0] = 0.0
        for iz in range(1, nz + 1):
            hflux[iz] = hflux[iz - 1] + dz[iz - 1] * src[iz - 1] / (4 * nc.pi)
        #
        # Now integrate back down
        #
        jmean[nz - 1] = hflux[nz] * np.sqrt(3.0)
        for iz in range(nz - 2, -1, -1):
            jmean[iz] = jmean[iz + 1] + 3 * hflux[iz + 1] * meanalphai[iz] * (z[iz + 1] - z[iz])
        #
        # Compute the temperature throughout the vertical structure
        #
        self.compute_temperature_from_radiation()

    def timestep_vert_rad_diffusion(self, dt):
        """
        For time-dependent radiative diffusion, use this subroutine. For the stationary
        case, use solve_vert_rad_diffusion().
        """
        assert hasattr(self, 'gamma'), "Error: The gamma (ratio of specific heats) must be specified."
        assert hasattr(self, 'mugas'), "Error: The mugas must be specified."
        if not hasattr(self, 'diff_jmean'):
            self.diff_jmean  = np.zeros_like(self.z)
        #
        # Get or compute variables
        #
        self.cv = nc.kk / ((self.gamma - 1.0) * self.mugas * nc.mp)
        jmean   = self.diff_jmean
        z       = self.z
        #
        # Compute the f_c factor of Kuiper, Klahr, Dullemond, Kley & Henning (2010)
        #
        fc    = 1.0 / (1.0 + self.cv * self.rhogas / (4 * nc.aa * self.tgas**3))
        #
        # Since the 1-D diffusion solver does not have a prefactor before the
        # time derivative, we simply modify the time step
        #
        cfcdt = nc.cc * fc[1:-1] * dt
        #
        # Compute the mean intensity from the local temperature
        #
        jmean[:] = (nc.ss / nc.pi) * self.tgas**4
        #
        # Set the diffusion coefficient at cell interfaces
        #
        rhkap = self.rhogas * self.mean_opacity_rosseland
        d     = 1.0 / (3.0 * 0.5 * (rhkap[1:] + rhkap[:-1]))
        #
        # Set the source term
        #
        s     = self.irrad_src / (4 * nc.pi)
        #
        # Boundary conditions
        #
        bcl   = (1.0, 0.0, 0.0, 0)
        bcr   = (d[-1], 1. / np.sqrt(3.0), 0.0, 0)
        #
        # Solve for one time step
        #
        v     = np.zeros_like(d)
        g     = np.ones_like(jmean)
        jmean[:] = solvediffonedee(z, jmean, v, d, g, s, bcl, bcr, dt=cfcdt, int=True)
        #
        # Recompute the temperature
        #
        self.compute_temperature_from_radiation()

    def compute_temperature_from_radiation(self):
        """
        Compute the gas/dust temperature from the direct and diffuse radiation field.
        """
        self.tgas = ((self.irrad_jmean + self.diff_jmean) * nc.pi / nc.ss)**0.25

    def compute_hflux_from_jmean(self):
        """
        For debugging purposes: re-compute H from J.
        """
        z     = self.z
        nz    = len(z)
        if not hasattr(self, 'diff_hflux'):
            self.diff_hflux    = np.zeros(nz + 1)
        hflux = self.diff_hflux
        jmean = self.diff_jmean
        rhok  = self.rhogas * self.mean_opacity_rosseland
        rhoki = 0.5 * (rhok[1:] + rhok[:-1])
        hflux[1:-1] = -(jmean[1:] - jmean[:-1]) / (z[1:] - z[:-1]) / (3 * rhoki)
        hflux[0]  = 0.0
        hflux[-1] = hflux[-2]

    def iterate_vertical_structure(self, tol=1e-10, maxiter=100, meanopacitymodel=None,
                                   include_curvature=True, add_selfgrav=False,
                                   iterate_selfgrav1d=False,dtgitermax=20,dtgerrtol=1e-4,
                                   incl_dustweight=True,comove_dust=True,comove_tracers=True):
        """
        Wrapper function that iterates the vertical structure hydrostatically and
        radiative transferly. The tolerance is given by tol. The maximum nr of
        iterations is given by maxiter.

        ARGUMENTS:
           include_curvature     If True, then include the 1/(1+(z/r)**2)**1.5 factor
                                 caused by the fact that the vertical model is exactly
                                 vertical while gravity goes with the spherical radius.
           add_selfgrav          If True, and self.selfgrav_fz[:] has been calculated (beforehand)
                                 then it will be added to the vertical forces.
           iterate_selfgrav1d    If True, it will also compute the self-gravity forces,
                                 but only in 1-D plane-parallel approximation. The 'real'
                                 self-gravity methods can be found in disk2d.py. 
           incl_dustweight:      If True, then include the dust weight in the hydrostatic equilibrium.
           comove_dust:          If True, then move the dust along with the gas as the latter finds a new
                                 hydrostatic equilibrium.
           dtgitermax:           (only if dust incl_dustweight): Max nr of iterations for
                                 dust weight inclusion.
           dtgerrtol:            (only if dust incl_dustweight): Error tolerance on dust-to-gas ratio.

        WARNING: If you include self-gravity, then please check if the disk is still
                 gravitationally stable, because otherwise the solution is unphysical.
        """
        if iterate_selfgrav1d: add_selfgrav=True
        if meanopacitymodel is None:
            assert hasattr(self, 'meanopacitymodel'), "Error: Must specify self.meanopacitymodel."
        else:
            self.meanopacitymodel = meanopacitymodel
        tprev = self.tgas.copy()
        for iter in range(maxiter):
            if iterate_selfgrav1d:
                self.compute_selfgravity_vertical_force_1d()
            self.compute_rhogas_hydrostatic_with_dust_and_tracers(include_curvature=include_curvature,
                                                                  add_selfgrav=add_selfgrav,
                                                                  dtgitermax=dtgitermax,
                                                                  dtgerrtol=dtgerrtol,
                                                                  incl_dustweight=incl_dustweight,
                                                                  comove_dust=comove_dust,
                                                                  comove_tracers=comove_tracers)
            self.compute_mean_opacity()
            self.irradiate_with_flaring_index()
            if hasattr(self, 'alphavisc'):
                self.compute_viscous_heating()
            self.solve_vert_rad_diffusion()
            self.compute_temperature_from_radiation()
            error = np.abs((self.tgas - tprev) / (self.tgas + tprev)).max()
            if error < tol:
                break
            tprev = self.tgas.copy()
        self.iter = iter

    def compute_vphi(self,dlnpdlnrc,add_selfgrav=False):
        """
        A realistic disk is not rotating perfectly keplerianly. Two important effects
        cause a deviation from Kepler rotation: (1) The radial pressure gradient and
        (2) the effect of the difference between cylindrical radius and spherical
        radius. The latter could also be interpreted as the Kepler velocity being
        not independent of z (in which case deviation 2 is not really a deviation
        from Kepler, but a change of Kepler with z). This method computes the phi-
        velocity (the azimuthal velocity) of the gas as a function of z, taking
        into account both effects. To do this, the radial pressure gradient has
        to be known. At every z the d\ln(p)/d\ln(r_cyl) at constant z has to be
        known (the argument dlnpdlnrc of this method). This can only be known
        in a truly 2-D (or 1+1-D) model of course. Or one can estimate it. For
        a disk with Sigma_gas ~ r^{-0.5}, T_mid ~ r^{-1.0}, for example (a disk
        with H_p ~ r) we have rho_gas_mid ~ r^{-1.5} and thus p_gas_mid ~ r^{-2.5}.
        At the midplane this would therefore give dlnpdlnrc=-2.5. But above
        the midplane this number will be less negative. The equation for v_phi
        is then:

           v_phi^2     G M_*           c_s^2 d ln(p)|
           ------- - --------- r_cyl - ----- -------|        = 0
            r_cyl    r_spher^3         r_cyl d r_cyl|z=const

        There may be points where v_phi^2<0. In these regions there cannot
        exist an equilibrium: the pressure gradient is too large, and the
        gas will move. Usually this occurs at extremely low density, meaning
        that these regions are anyway as good as empty. Therefore in these
        regions we simply set v_phi=0.

        If add_selfgrav==True, then also the cylindrical-radial component of
        the self-gravity force is added (assuming it has been computed 
        beforehand by the 2-D self-gravity method of disk2d.py). 
        """
        cs2   = nc.kk * self.tgas / ( self.mugas * nc.mp )
        vphi2 = self.omk_full**2 * self.r**2 + cs2 * dlnpdlnrc
        if add_selfgrav and hasattr(self,'selfgrav_fr'):
            vphi2 -= self.selfgrav_fr * self.r
        vphi2[vphi2<0] = 0
        self.vphi = np.sqrt(vphi2)

    def get_cumuldens(self,normalize=False):
        """
        Compute the int_0^z rho(z')dz' for all points in the z grid.
        Cell centered.
        """
        nz      = len(self.z)
        sig    = np.zeros(nz)
        for iz in range(1,nz):
            rho     = 0.5 * ( self.rhogas[iz] + self.rhogas[iz-1] )
            sig[iz] = sig[iz-1] + ( self.z[iz] - self.z[iz-1] ) * rho
        if np.abs(2 * sig[-1] / self.siggas - 1.0) > 1e-3:
            raise ValueError('Error in get_cumuldens(): Column density of rhogas unequal to siggas.')
        if normalize:
            sig = sig / sig[-1]
        return sig

    def add_dust(self, agrain, dtg=0.01, xigrain=3.0, grainmodel=None):
        """
        This routine adds a dust component to the model. To compute the dynamic
        properties of the dust, we use the grainmodel.py module.

        Arguments:
        ----------

        agrain : float
            particle size in cm

        Keywords:
        ---------

        dt2 : float
            dust to gas mass ratio

        xigrain : float
            material density in [g/cm^3]

        grainmodel : GrainModel
            if given, this will be linked to the component, otherwise
            a new instance is created

        Output:
        -------
        Adds a new dust component to the list self.dust
        """
        dust = DiskVerticalComponent(self, massfraction=dtg, agrain=agrain,
                                 xigrain=xigrain, grainmodel=grainmodel)
        if not hasattr(self, 'dust'):
            self.dust = []
        self.dust.append(dust)


class DiskVerticalComponent(object):
    """
    To allow the disk to contain one or more dust or chemical components, this class contains
    all the methods needed for moving such a component vertically (e.g. dust drift, mixing etc).
    """
    def __init__(self, diskverticalmodel, massfraction, agrain=None, xigrain=None, grainmodel=None):
        """
        Arguments:
        ----------

        diskverticalmodel : instance of DiskVerticalModel
            Link to the parent disk vertical structure model

        massfraction : float | array
            either a float or array of size diskverticalmodel.z.size
            this will initialize the density of this component to be
            massfraction * gas density
        """
        self.diskverticalmodel = diskverticalmodel     # Link to the parent disk vertical structure model
        self.z                 = diskverticalmodel.z   # Link vertical grid to disk model for convenience
        self.dz                = diskverticalmodel.dz  # Link vertical grid to disk model for convenience
        # nz = len(self.z)
        self.rho               = diskverticalmodel.rhogas * massfraction
        if xigrain is not None:
            assert np.isscalar(agrain), "xigrain must be a scalar"
            self.xigrain = xigrain
        else:
            self.xigrain = 3.0         # Default dust material density
        if agrain is not None:
            assert np.isscalar(agrain), "agrain must be a scalar"
            self.agrain  = agrain
            self.compute_tstop_from_agrain()

            if grainmodel is None:
                self.grain = GrainModel(agrain=agrain, xigrain=xigrain)
            else:
                assert agrain == grainmodel.agrain, "agrain must be the same as in grainmodel"
                self.grain = grainmodel

    def compute_surfacedensity(self):
        """
        Integrates the density vertically (multiplied by 2, because of the two sides
        of the disk) to obtain the surface density.
        """
        self.sigma = 2*(self.dz*self.rho).sum()
        
    def compute_mgrain_from_agrain(self):
        """
        Compute the grain mass everywhere.
        """
        assert hasattr(self, 'agrain'), "Error: no agrain present; cannot compute grain mass"
        assert hasattr(self, 'xigrain'), "Error: no xigrain present; cannot compute grain mass"
        nz = len(self.z)
        if not hasattr(self, 'mgrain'):
            self.mgrain  = np.zeros(nz)
        agrain  = np.zeros(nz) + self.agrain
        xigrain = np.zeros(nz) + self.xigrain
        self.mgrain[:] = (4 * nc.pi / 3.) * xigrain * agrain**3

    def compute_tstop_from_agrain(self, dv=1e3):
        """
        Compute the stopping time of the grain at each position.
        """
        nz    = len(self.z)
        if not hasattr(self, 'tstop'):
            self.tstop  = np.zeros(nz)
        self.compute_mgrain_from_agrain()
        nz           = len(self.z)
        mgrain       = self.mgrain
        agrain       = self.agrain
        xigrain      = self.xigrain
        for iz in range(nz):
            d        = GrainModel(agrain=agrain, xigrain=xigrain)
            d.mgrain = mgrain
            rhogas   = self.diskverticalmodel.rhogas[iz]
            tgas     = self.diskverticalmodel.tgas[iz]
            d.compute_tstop(rhogas, tgas, dv)
            self.tstop[iz] = d.tstop

    def compute_vsett_and_dmix(self):
        """
        Compute the settling velocity and the turbulent diffusion coefficient
        for this species.
        """
        z       = self.z
        nz      = len(z)
        omk     = self.diskverticalmodel.omk_midpl
        tgas    = self.diskverticalmodel.tgas
        mugas   = self.diskverticalmodel.mugas
        cs2     = nc.kk * tgas / (mugas * nc.mp)
        tstop   = self.tstop
        St      = omk * tstop
        self.vsett = np.maximum(-z * omk**2 * tstop, -omk * z)
        if hasattr(self.diskverticalmodel, 'alphamix'):
            dmix = self.diskverticalmodel.alphamix * cs2 / omk / self.diskverticalmodell.Sc
        else:
            self.diskverticalmodel.compute_local_shear_viscosity()
            dmix = self.diskverticalmodel.nuvisc / self.diskverticalmodel.Sc
        self.dmix = dmix + np.zeros(nz)
        self.dmix[:] *= 1.0 / (1.0 + St**2)

    def timestep_settling_mixing(self, dt):
        """
        Do a time step of the settling mixing equation.
        """
        self.compute_tstop_from_agrain()
        self.compute_vsett_and_dmix()
        z       = self.z
        nz      = len(z)
        rhogas  = self.diskverticalmodel.rhogas
        s       = np.zeros(nz)
        vdusti  = 0.5 * (self.vsett[1:] + self.vsett[:-1])
        diffi   = 0.5 * (self.dmix[1:]  + self.dmix[:-1])
        vdusti[-1] = 0.
        vdusti[0]  = 0.
        #bcl     = (1., 0., 0., 1)
        bcl     = (1., 0., 0., -1)
        bcr     = (1., 0., 0., 0)
        self.rho[:] = solvediffonedee(z, self.rho, vdusti, diffi, rhogas, s, bcl, bcr, dt=dt, int=True, upwind=True)

    def compute_settling_mixing_equilibrium(self, n=200, update_variables=True):
        """
        Solve for the stationary settling-mixing solution, see Fromang & Nelson,
        2009, A&A 496, 597. The settling-mixing equilibrium can be written as
                       z
           /  c   \    /   v
        ln | ---- | =  | -----  dz
           \  c0  /    /   D
                       0
        The settling velocity and diffusivity are updated on the fly unless this
        is specifically not wanted (via setting update_variables=False).

        The integral will be solved on a different grid than defined in the
        model. This should increase the accuracy for low resolution. The result
        is normalized.

        Keywords:
        ---------

        n : int
            number of vertical grid points on which to solve the integral
            equation. To force the solution to be carried out on the
            original grid, use n=0

        update_variables : bool
            by default, `compute_tstop_from_agrain` and `compute_vsett_and_dmix`
            are called before the calculation. To avoid that, set
            `update_variables=False`

        Output:
        ------

        updates self.rho
        """
        from scipy.integrate import cumtrapz

        if update_variables:
            self.compute_tstop_from_agrain()
            self.compute_vsett_and_dmix()

        # get the surface density for normalization

        sig_d = self.diskverticalmodel.vertically_integrate(self.rho)

        diff  = self.dmix + 1e-100
        integrand = self.vsett / diff

        if n > 0:

            # solve it on a finer grid

            z         = np.logspace(np.log10(self.z[1] / 100), np.log10(self.z[-1]), n - 1)
            z         = np.hstack((0, z))
            rho_g     = 10.**np.interp(z, self.z, np.log10(self.diskverticalmodel.rhogas))
            integrand = np.interp(z, self.z, integrand)
            integrand[np.isnan(integrand)] = 0.0
            integrand[np.isinf(integrand)] = 0.0

        else:

            # solve it on the original grid

            z     = self.z
            rho_g = self.diskverticalmodel.rhogas

        # solve the integral and calculate the solution

        cumintegral = cumtrapz(integrand, x=z, initial=0)
        rho = rho_g * np.exp(np.maximum(cumintegral, -700))
        rho[rho < 1e-100] = 1e-100

        # interpolate back if needed

        if n > 0:
            rho = 10.**np.interp(self.z, z, np.log10(rho))

        # normalize to the dust surface density

        self.rho = rho / self.diskverticalmodel.vertically_integrate(rho) * sig_d
