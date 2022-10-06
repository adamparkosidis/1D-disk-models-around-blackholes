import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy import optimize
from . import natconst as nc
from .grainmodel import GrainModel
from .solvediffonedee import solvediffonedee, getfluxonedee
from .meanopacity import evaluate_meanopacity
import copy
import warnings
from .brent_array import brent_array

from scipy.integrate import solve_ivp

try:
    from numba import njit, jit
except ImportError:
    warnings.warn('numba not available, opacity calculation will be very slow')

    def njit(ob):
        return ob

    jit = njit


class DiskRadialModel(object):
    """
    The basic disk model of the DISKLAB package.

    ARGUMENTS (all optional, most with default values):
    mstar             Stellar mass [g]
    lstar             Stellar luminosity [erg/s]
    tstar             Stellar temperature [K]
    rin               Inner disk radius [cm]
    rout              Outer disk radius [cm]
    nr                Nr of grid points
    mdisk             Mass of the disk [g]
    plsig             Powerlaw of surface density
    flang             Flaring irradiation angle [radian]
    alpha             Alpha value for turbulence
    Q                 Toomre Q value (from which surface density is computed)
    mdot              Accretion rate [g/s] (from which surface density is computed)
    rgrid             If set to an array, then use this array as the radial
                      grid (ignoring rin, rout, nr).
    tbg               The background temperature [K] (Default: cosmic background)
    #opedit:
    mu                mean molecular weight (defaults to 2.3) 

    Note: Either specify mdisk or mdot or Q, not more than one.
    Note: You can also create a Lynden-Bell & Pringle model, but since that
    requires more than just 1 parameter, this requires a separate call to
    make_disk_from_lyndenbellpringle().

    You can add a dust component with add_dust()

    EXAMPLES OF VARIOUS DISK MODELS:

    from disklab.diskradial import *
    from disklab.natconst import *
    import matplotlib.pyplot as plt
    a=DiskRadialModel(mdot=1e-8*MS/year)
    b=DiskRadialModel(mdisk=1e-2*MS)
    c=DiskRadialModel(qtoomre=2.0)
    d=DiskRadialModel()
    d.make_disk_from_lbp_alpha(1e-2*MS,1*AU,1e-2,1e5*year)
    plt.plot(a.r/au,a.sigma,label=r'$\dot M=10^{-8}M_\odot/\mathrm{year}$')
    plt.xlabel(r'$r\; \mathrm{[au]}$')
    plt.ylabel(r'$\Sigma\; \mathrm{[g/cm^2]}$')
    plt.xscale('log')
    plt.yscale('log')
    plt.plot(b.r/au,b.sigma,label=r'$M_{\mathrm{disk}}=10^{-2}M_\odot$')
    plt.plot(c.r/au,c.sigma,label=r'$Q=2$')
    plt.plot(d.r/au,d.sigma,label=r'$\mathrm{LBP}:\quad \alpha=0.01, t=10^5\mathrm{year}$')
    plt.legend(loc='lower left')
    plt.show()

    EXAMPLE OF ADDING DUST AND SETTING ALPHA:

    from disklab.diskradial import *
    from disklab.natconst import *
    import matplotlib.pyplot as plt
    a=DiskRadialModel(mdot=1e-8*MS/year,alpha=1e-3)
    a.add_dust(agrain=1e-1)   # mm dust
    a.compute_qtoomre()
    a.dust[0].compute_qtoomre()
    plt.figure()
    plt.plot(a.r/au,a.rhomid,label='Gas')
    plt.plot(a.r/au,a.dust[0].rhomid,label='Dust')
    plt.xlabel(r'$r\; \mathrm{[au]}$')
    plt.ylabel(r'$\rho(z=0)\; \mathrm{[g/cm^3]}$')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(loc='lower left')
    plt.figure()
    plt.plot(a.r/au,a.qtoomre,label='Gas')
    plt.plot(a.r/au,a.dust[0].qtoomre,label='Dust')
    plt.xlabel(r'$r\; \mathrm{[au]}$')
    plt.ylabel(r'$Q$')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(loc='upper right')
    plt.figure()
    plt.plot(a.r/au,a.hp/a.r,label='Gas')
    plt.plot(a.r/au,a.dust[0].hp/a.r,label='Dust')
    plt.xlabel(r'$r\; \mathrm{[au]}$')
    plt.ylabel(r'$h/r$')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(loc='upper right')
    plt.show()

    EXAMPLE OF GRID REFINEMENT:

    nrin   = 200
    nrmid  = 200
    nrout  = 200
    rin    = 1*au
    r1     = 40.*au
    r2     = 60.*au
    rout   = 1e3*au
    rinner = rin * (r1/rin)**np.linspace(0,1,nrin)
    rmid   = r1 * (r2/r1)**np.linspace(0,1,nrmid)
    router = r2 * (rout/r2)**np.linspace(0,1,nrout)
    r      = np.hstack((rinner[:-1],rmid[:-1],router))
    d      = DiskRadialModel(rgrid=r)

    """

    def __init__(self, mstar=nc.MS, lstar=0, tstar=nc.TS, rin=0.1 * nc.au,
                 rout=200 * nc.au, nr=100, rgrid=None, mdisk=None, plsig=-1,
                 flang=0.05, alpha=1e-2, qtoomre=None, mdot=None, hpr=0.01,
                 meanopacitymodel=None, tbg=0., mu=0.615):
        self.mstar = mstar
        self.lstar = lstar
        self.tstar = tstar
        self.rstar = nc.RS * (lstar / nc.LS)**0.5 * (nc.TS / tstar)**0.25
        self.tbg   = tbg
        self.mu    = mu
        if rgrid is None:
            self.r = rin * (rout / rin)**np.linspace(0., 1., nr)
        else:
            self.r = rgrid.copy()
        self.dsurf  = self.get_dsurf()
        self.alpha  = alpha
        self.flang  = flang
        self.plsig  = plsig
        self.Sc     = 1.0      # The Schmidt number of (turbulent) gas viscosity / gas diffusion
        self.sigmin = 1e-30
        assert self.r.min() > self.rstar, "Inner disk edge inside star"
        self.omk   = (nc.GG * self.mstar / self.r**3)**0.5
        if hpr is None:
            self.compute_disktmid()   # Self-consistent T_mid computation
        else:
            self.hp   = hpr * self.r
            self.cs   = self.hp * self.omk
            self.tmid = self.mu * nc.mp * self.cs**2 / nc.kk
        if mdisk is not None:
            self.make_disk_from_m_pl(mdisk)
        if mdot is not None:
            self.make_disk_from_mdot(mdot)
        if qtoomre is not None:
            self.make_disk_from_toomre(qtoomre)
        if meanopacitymodel is not None:
            self.meanopacitymodel = meanopacitymodel
            self.compute_mean_opacity()

    def update_stellar_mass(self,mstar):
        """
        If you change the mstar, you must recompute everything that
        depends on the stellar mass. So when adapting mstar, always
        use this function.
        """
        self.mstar = mstar
        self.omk   = (nc.GG * self.mstar / self.r**3)**0.5
        if hasattr(self, 'cs'):
            self.hp = self.cs / self.omk

    def get_ri(self,full=False):
        """
        Compute the interface radii.
        """
        ri = 0.5 * (self.r[1:] + self.r[:-1])
        if full:
            ri = np.hstack((self.r[0],ri,self.r[-1]))
        return ri

    def get_dsurf(self):
        """
        Compute the surface area belonging to each annulus
        """
        ri    = self.get_ri(full=True)
        dsurf = np.pi*(ri[1:]**2-ri[:-1]**2)
        return dsurf


    # -------------------------------------------------------------------------
    #                     STANDARD DISK MODELS
    #
    # Here are a set of standard analytic disk models from the literature.
    # Several are so standard that no reference needs to be given.
    # -------------------------------------------------------------------------

    def make_disk_from_powerlaw(self, sig0, r0, plsig):
        """
        Make a powerlaw model of the gas surface density of
        the disk with parameters sig0, r0 and plsig.
        """
        self.plsig    = plsig
        self.sig0     = sig0
        self.r0       = r0
        self.sigma    = sig0 * (self.r / r0)**plsig
        self.compute_mass()
        self.compute_rhomid_from_sigma()

    def make_disk_from_m_pl(self, mdisk, plsig=None, rdisk=None):
        """
        Make a model of the gas surface density of the disk.
        Here the model is given by mdisk and the surface
        density powerlaw index self.plsig. This routine will
        compute self.sigma such that the mass is as specified.
        """
        if plsig is not None:
            self.plsig = plsig
        self.sigma   = self.r**self.plsig
        if rdisk is not None:
            self.sigma[self.r > rdisk] = 0.0
        self.compute_mass()
        self.sigma  *= mdisk / self.mass
        if rdisk is not None:
            self.sigma[self.r > rdisk] = self.sigmin
        self.compute_mass()
        self.compute_rhomid_from_sigma()

    def make_disk_from_toomre(self, Q):
        """
        Make a model of the gas surface density of the disk.
        Here the model is given by setting the surface density
        such that the Toomre Q value has the value given as
        argument. This routine will compute self.sigma, and
        if self.hp is set, also the midplane density.
        """
        self.sigma  = self.omk * self.cs / (nc.pi * nc.GG * Q)
        self.compute_mass()
        self.compute_rhomid_from_sigma()

    def make_disk_from_mdot(self, mdot):
        """
        Make a model of the gas surface density of the disk.
        Here the model is set by the global accretion rate mdot
        [g/s] (assumed to be constant with radius and time).
        """
        assert hasattr(self, 'cs'), "Error: sound speed not set"
        assert hasattr(self, 'alpha'), "Error: alpha not set"
        self.mdot   = mdot
        alpha       = self.alpha
        omk         = self.omk
        cs          = self.cs
        nu          = alpha * cs * cs / omk
        self.sigma  = mdot / (3 * nc.pi * nu)
        self.compute_mass()
        self.compute_rhomid_from_sigma()

    def make_disk_from_lyndenbellpringle(self, M0, R0, nu0, gam, time):
        """
        Make a model of the gas surface density of the disk.
        Here the model is set by the Lynden-Bell & Pringle solution.
        I follow here Lodato, Scardoni, Manara & Testi (2017)
        MNRAS 472, 4700, their Eqs. 2, 3 and 4.

        ARGUMENTS:
          M0      = Initial disk mass [g]
          R0      = Initial disk radius [cm]
          nu0     = Viscosity nu at R0 [cm^2/s]
          gam     = The gamma coefficient of the LBP model
          time    = Time after start [s]
        """
        r        = self.r
        nu       = nu0 * (r / R0)**gam                 # Eq. 2
        tnu      = R0**2 / (3 * (2. - gam)**2 * nu0)   # Eq. 4
        T        = 1.0 + time / tnu                    # Above Eq. 4
        eta      = (2.5 - gam) / (2. - gam)            # Below Eq. 3
        sigma    = (M0 / (2 * nc.pi * R0**2)) * (2. - gam) * \
                   (R0 / r)**gam * T**(-eta) * np.exp(-(r / R0)**(2. - gam) / T)  # Eq. 3
        self.M0     = M0
        self.R0     = R0
        self.nu     = nu
        self.tnu    = tnu
        self.gam    = gam
        self.time   = time
        self.T      = T
        self.sigma  = sigma
        self.compute_mass()
        self.sigma[self.sigma < self.sigmin] = self.sigmin
        self.compute_rhomid_from_sigma()

    def make_disk_from_lbp_alpha(self, M0, R0, alpha, time):
        """
        Same as make_disk_from_lyndenbellpringle(), but now assuming a constant
        alpha viscosity model. That is: both nu0 and gamma are internally
        computed.

        ARGUMENTS:
          M0      = Initial disk mass [g]
          R0      = Initial disk radius [cm]
          alpha   = The Shakura and Sunyaev alpha value
          time    = Time after start [s]

        NOTE: self.cs must be a powerlaw function!
        """
        assert hasattr(self, 'cs'), "Requires cs to compute viscosity"
        assert hasattr(self, 'omk'), "Requires omk to compute viscosity"
        self.alpha = alpha
        r    = self.r
        cs   = self.cs
        omk  = self.omk
        nu   = alpha * cs * cs / omk
        gam  = (np.log(nu[-1]) - np.log(nu[0])) / (np.log(r[-1]) - np.log(r[0]))

        f    = interp1d(r, nu)
        nu0  = f([R0])[0]
        self.make_disk_from_lyndenbellpringle(M0, R0, nu0, gam, time)

    def make_disk_from_simplified_lbp(self, Sigc, Rc, gam):
        """
        Make a model of the gas surface density of the disk.
        Here the model is set by the simplified LBP-like powerlaw
        disk with an exponential cut-off with critical radius Rc.
        I follow here Takahashi & Inutsuka (2016)
        AJ 152, 184, their Eq. 18.

        ARGUMENTS:
          Sigc    = Normalization constant [g/cm^2]
          Rc      = Critical disk radius [cm]
          gam     = The gamma coefficient of the LBP model
        """
        r           = self.r
        sigma       = Sigc * (Rc / r)**gam * np.exp(-(r / Rc)**(2. - gam))
        self.Sigc   = Sigc
        self.Rc     = Rc
        self.sigma = sigma
        self.sigma[self.sigma < self.sigmin] = self.sigmin
        self.compute_mass()
        self.compute_rhomid_from_sigma()

    # -------------------------------------------------------------------------
    #                 STANDARD METHODS FOR MISCELLANEOUS
    #
    # Now follow a series of standard things for the DiskRadialModel class.
    # -------------------------------------------------------------------------

    def compute_mean_opacity(self, meanopacitymodel=None):
        """
        To compute the radiative heating and cooling of a protoplanetary disk, we need
        the opacity of the disk material. This is most likely the dust, but it could also
        be gas opacity for the hot regions. Or you could decide to specify a simple opacity
        law independent of the dust species in the disk.

        The method compute_mean_opacity() creates and fills two arrays:

           self.mean_opacity_planck[0:nr]
           self.mean_opacity_rosseland[0:nr]

        which are the Planck-mean and Rosseland-mean opacities respectively. They are
        normalized to the *gas* density, i.e. the Rosseland vertical optical depth would be:

           tau[0:nr] = self.sigma[0:nr] * self.mean_opacity_rosseland[0:nr]

        The normalization to the gas density (instead of the dust density) is simply
        because "the" gas of the disk is just a single entity, whereas DISKLAB allows
        multiple dust species. The mean opacity is thus "cross section per gram of gas",
        but it will (or may) include any opacity bearing material suspended in the gas.
        So for a typical dust opacity we get that the mean opacity is the dust-to-gas
        ratio times the dust opacity.

        Note that if the temperature of the disk changes, you will have to call this
        method compute_mean_opacity() again to recompute the opacity arrays. This will
        typically happen when doing a time step of the disk evolution, where the surface
        density and temperature change with time.

        Note also that if the opacities depend on a r-dependent distribution of disk
        components (e.g. on dust drift or on an r-dependent disk chemical composition),
        then one also has to re-call this method every time this radial distribution
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
            self.mean_opacity_planck = np.zeros_like(self.sigma)
        if not hasattr(self, 'mean_opacity_rosseland'):
            self.mean_opacity_rosseland = np.zeros_like(self.sigma)
        #
        # Get meanopacitymodel
        #
        if meanopacitymodel is None:
            assert hasattr(self, 'meanopacitymodel'), 'To install the mean opacities you must give an opacity model.'
            meanopacitymodel = self.meanopacitymodel
        #
        # Check if meanopacitymodel is a scalar
        #
        if np.isscalar(meanopacitymodel):
            meanopacitymodel = [meanopacitymodel]
        self.meanopacitymodel = meanopacitymodel
        #
        # Make sure that the midplane density is up-to-date
        #
        self.compute_cs_and_hp()
        self.compute_rhomid_from_sigma()
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
                    if hasattr(d, 'rhomid'):
                        rhodust.append(d.rhomid)
                    else:
                        rhodust.append(None)
                    if hasattr(d, 'grain'):
                        grain.append(d.grain)
                    else:
                        grain.append(None)
        #
        # Check if we have the midplane gas density and temperature available
        #
        if hasattr(self, 'rhomid'):
            rhogas = self.rhomid
        else:
            rhogas = None
        if hasattr(self, 'tmid'):
            tgas = self.tmid
        else:
            tgas = None
        #
        # Now handle the different meanopacitymodels:
        #
        if np.isscalar(rhogas):  rhogas  = np.array([rhogas])
        if np.isscalar(tgas):    tgas    = np.array([tgas])
        if rhodust is not None:
            if np.isscalar(rhodust): rhodust = np.array([rhodust])
        meanopac = evaluate_meanopacity(meanopacitymodel, rhogas, tgas, rhodust=rhodust, grain=grain)
        self.mean_opacity_planck[:]    = meanopac['planck']
        self.mean_opacity_rosseland[:] = meanopac['rosseland']

    def compute_disktmid(self, vischeat=False, fixmeanopac=False,
                         keeptvisc=False, fixmdot=False, simple=False,
                         alphamodel=True):
        """
        Compute the midplane temperature based on the simple
        irradiation recipe in which half of the irradiated
        energy is radiated away, half is heating the midplane
        of the disk. Will compute self.tmid, the midplane
        temperature, self.cs, the isothermal sound speed,
        and self.hp, the vertical scale height.

        Note that there will always be some irradiation by the background
        radiation field (temperature: self.tbg), which by default it set
        to 2.725 K, the cosmic background temperature.

        If vischeat==True, then also include the viscous
        heating. But for that we need sigma and mean_opacity_rosseland.

        NOTE: Since this method uses the Rosseland mean opacity array
              self.mean_opacity_rosseland[:], if you are not sure if this
              is still up-to-date with the current surface density and
              temperature, then better call self.compute_mean_opacity()

        If keeptvisc==True but vischeat==False, then we do
        not calculate the self.tvisc, but we keep the
        self.tvisc (i.e. we do not put it to zero).

        The fixmeanopac, fixmdot, alphamodel and simple flags is only meant for 
        passing to the call to self.solve_viscous_heating_globally().
        See documentation there for their meaning.

        COMPUTES:
        self.tmid   = Midplane temperature in [K]
        self.cs     = Isothermal sound speed [cm/s]
        self.hp     = Pressure scale height [cm]
        self.qirr   = Irradiative heating rate of the disk [erg/cm^2/s]
        self.qvisc  = (if vischeat==True) Viscous heating rate [erg/cm^2/s]
        """
        nr        = len(self.r)
        flux      = self.lstar / (4 * nc.pi * self.r**2)
        self.qirr = flux * self.flang   # Factor 2 for two sides, factor 0.5 for half irradiated down
        self.tirr = ( (0.5 * self.qirr / nc.ss ) + self.tbg**4 )**0.25  # Factor 0.5 because cooling is two-sided
        if vischeat:
            assert hasattr(self, 'sigma'), "Error in computing Tmid with visc heating: need sigma"
            #assert hasattr(self, 'dust'), "Error in computing Tmid with visc heating: need dust"
            assert hasattr(self, 'mean_opacity_rosseland'), "Error in computing Tmid with visc heating: need mean_opacity_rosseland."
            self.solve_viscous_heating_globally(fixmeanopac=fixmeanopac,fixmdot=fixmdot,simple=simple,alphamodel=alphamodel)
        else:
            if (not keeptvisc) or (not hasattr(self, 'tvisc')):
                self.tvisc = np.zeros(nr)
        self.tmid = (self.tirr**4 + self.tvisc**4)**0.25
        self.compute_cs_and_hp()

    def get_total_viscous_heating_minus_cooling(self,fixmeanopac=False,log=False,
                                                heating=True,cooling=True,alphamodel=True):
        """
        This is a helper function for solve_viscous_heating_globally().
        Given the self.tirr and self.tvisc it returns the Q_visc - Q_cool. This
        is used for bisection for finding the viscous temperature in 
        regions where the opacity declines strongly with temperature
        (for instance near the dust sublimation temperature).
        """
        self.tmid = (self.tirr**4 + self.tvisc**4)**0.25
        self.compute_cs_and_hp()
        self.compute_rhomid_from_sigma()
        if alphamodel:
            self.compute_nu()
        if not fixmeanopac:
            self.compute_mean_opacity()
        tau         = self.sigma * self.mean_opacity_rosseland
        dummy       = 1.-np.exp(-2.*tau)
        mask        = tau<0.5e-6
        dummy[mask] = 2*tau[mask]
        factorinv   = dummy/(0.5*tau+1.)
        qvisc       = (9. / 4.) * self.sigma * self.nu * self.omk**2
        qcool       = 2*nc.ss*self.tvisc**4*factorinv
        if log:
            assert len(np.where(qvisc<=0)[0])==0, 'Error: Zero Q_visc found. Forbidden for log mode.'
            assert len(np.where(qcool<=0)[0])==0, 'Error: Zero Q_cool found. Forbidden for log mode.'
            qvisc = np.log10(qvisc)
            qcool = np.log10(qcool)
        answer  = 0.
        if heating:
            answer += qvisc
        if cooling:
            answer -= qcool
        return answer
        
    def solve_viscous_heating_globally(self, ttol=1e-2, qtol=1e-4, nitermax=100,
                                       fixmeanopac=False, simple=True,
                                       fixmdot=False, alphamodel=True):
        """
        Solve the midplane tempeature due to viscous heating (only).

        ARGUMENTS:
         ttol            The tolerance for the convergence (in Kelvin)
                         (only relevant when simple==True)
         qtol            The tolerance for the convergence (in log(Q_tot))
                         (only relevant when simple==False)

        OPTIONAL KEYWORDS:
         nitermax        The number of iterations before giving up.
         fixmeanopac     If True, then do not update the Rosseland mean opacity
                         during the iteration (for testing purposes only)
         simple          If True:  Compute midplane temperature by simple
                                   iteration of the Tmid-formula keeping 
                                   opacity constant, but updating opacity
                                   after each iteration. This sometimes does
                                   not converge, especially when the opacity
                                   drops with temperature (i.e. when dust 
                                   evaporates)
                         If False: Compute midplane temperature by using 
                                   Brent's method. This is much more robust
                                   than the simple method. Is preferred when
                                   dust evaporation plays a role.
         fixmdot         If True:  (and if simple==False) then instead of
                                   keeping the Sigma constant during the
                                   root-finding, we keep the accretion rate
                                   Mdot constant, though only in a simplified
                                   way: we vary Sigma ~ 1/T, so that Sigma*T
                                   remains constant. But at the end we 
                                   reset Sigma. It is just a trick to keep
                                   things stable.
        """
        assert hasattr(self, 'mean_opacity_rosseland'), 'Can only solve viscous heating if mean opacities are installed'
        nr        = len(self.r)
        if hasattr(self,'tvisc'):
            tviscorig  = self.tvisc.copy()
        else:
            tviscorig  = None
            self.tvisc = np.zeros(nr)
        #
        # Choose method
        #
        if simple:
            #
            # Simple iteration method
            #
            assert not fixmdot, 'Error: fixmdot incompatible with simple'
            for iter in range(nitermax):
                tviscold   = self.tvisc.copy()
                if alphamodel:
                    self.compute_nu()
                self.qvisc = (9. / 4.) * self.sigma * self.nu * self.omk**2
                tau        = self.sigma * self.mean_opacity_rosseland
                dummy      = 1.-np.exp(-2.*tau)
                mask       = tau<0.5e-6
                dummy[mask]= 2*tau[mask]
                factor     = (0.5*tau+1.)/dummy
                #opedit: different to FKR...
                #self.tvisc = ( self.qvisc * factor / (2*nc.ss) )**0.25
                # according to FKR:
                self.tvisc = (self.qvisc/2. * 3./4. * tau / nc.ss)**0.25
                tvdiff     = np.abs(self.tvisc - tviscold)
                diff       = tvdiff.max()
                self.tmid  = (self.tirr**4 + self.tvisc**4)**0.25
                self.compute_cs_and_hp()
                if not fixmeanopac:
                    self.compute_mean_opacity()
                if diff < ttol:
                    return
        else:
            #opedit: uses somewhat different viscous heating to FKR.
            # you should use 'simple' which I checked. 
            # I'll put a warning here:
            print('Warning: Brents method does not 100% agree with the book of FKR use "simple=True" instead')
            #
            # Brent's method
            #
            # We use the following function for that:
            #
            def qtotal(tvisc,aux=None):
                d       = aux["d"]
                fixmdot = aux["fixmdot"]
                if "fixmeanopac" in aux.keys():
                    fixmeanopac = aux["fixmeanopac"]
                else:
                    fixmeanopac = False
                d.tvisc = tvisc.copy()
                if fixmdot:
                    d.tmid     = (d.tirr**4 + d.tvisc**4)**0.25
                    sigma_orig = aux["sigma_orig"]
                    tmid_orig  = aux["tmid_orig"]
                    d.sigma    = sigma_orig * (tmid_orig/d.tmid)
                qtot    = d.get_total_viscous_heating_minus_cooling(fixmeanopac=fixmeanopac,log=True)
                return qtot
            #
            # Then let's find an interval in temperature that contains a solution
            # (or at least one solution). The lower we choose fmax, the lower the
            # chance that we accidently bracket more than one solution.
            #
            # IMPORTANT NOTE: We will use the current (i.e. previous) values
            #                 of self.tvisc (if existent) as starting value.
            #
            aux         = {"d":self,"fixmeanopac":fixmeanopac,"fixmdot":fixmdot}
            if fixmdot:
                aux["sigma_orig"] = self.sigma.copy()
                aux["tmid_orig"]  = self.tmid.copy()
            fmax        = 2.
            if not hasattr(self,'tvisc'):
                self.tvisc = np.zeros_like(self.r)
            if len(np.where(self.tvisc<=0)[0])>0:
                if alphamodel:
                    self.compute_nu()
                self.qvisc = (9. / 4.) * self.sigma * self.nu * self.omk**2
                self.tvisc = ( self.qvisc / (2*nc.ss) )**0.25  # Lowest estimate
            qtot        = qtotal(self.tvisc,aux=aux)
            fact        = 10**np.abs(qtot)
            fact[fact>fmax] = fmax
            tlow        = self.tvisc / fact
            thigh       = self.tvisc * fact
            qlow        = qtotal(tlow,aux=aux)
            qhigh       = qtotal(thigh,aux=aux)
            dummy       = qlow*qhigh
            mask        = dummy>0
            ii          = np.where(mask)[0]
            iter        = 0
            nitermax    = 100
            while len(ii)>0 and iter<nitermax:
                tlow[ii]       /= fact[ii]
                thigh[ii]      *= fact[ii]
                qlow            = qtotal(tlow,aux=aux)
                qhigh           = qtotal(thigh,aux=aux)
                dummy[ii]       = qlow[ii]*qhigh[ii]
                mask            = dummy>0
                ii              = np.where(mask)[0]
                iter           += 1
            assert iter<nitermax, "No convergence in finding temperature range"
            #
            # Now with this interval use Brent's method to find the solution
            #
            temp,qtot,info = brent_array(tlow,thigh,qtotal,qtol,aux=aux,returninfo=True)
            self.visc      = temp
            self.tmid      = (self.tirr**4 + self.tvisc**4)**0.25
            if fixmdot:
                self.sigma = aux["sigma_orig"]
            self.brentinfo = info
            if info["converged"]:
                return
        #
        # If we arrive here, then we have not converged
        #
        raise StopIteration('Viscous heating did not converge')

    def compute_cs_and_hp(self):
        """
        Compute from the midplane temperature the isothermal
        sound speed and the disk vertical scale height.
        """
        self.cs   = (nc.kk * self.tmid / (2.3 * nc.mp))**0.5
        self.hp   = self.cs / self.omk

    def compute_rhomid_from_sigma(self):
        """
        Compute the midplane gas density from the surface gas density
        given the vertical scale height hp.
        """
        assert hasattr(self, 'hp'), "Cannot compute midplane rhomid without hp"
        # opedit:
        # inconsistent with FKR
        #self.rhomid  = self.sigma / (self.hp * (2 * nc.pi)**0.5)
        self.rhomid  = self.sigma / self.hp

    def compute_nu(self):
        """
        Compute the viscous coefficient.
        """
        self.nu = self.alpha * self.cs * self.cs / self.omk

    def compute_mass(self):
        """
        Compute the disk mass by numeric integration. Will be put in self.mdisk.
        """
        ds = nc.pi * (self.r[1:]**2 - self.r[:-1]**2)
        sg = 0.5 * (self.sigma[1:] + self.sigma[:-1])
        dm = sg * ds
        self.mass = dm.sum()

    def compute_qtoomre(self):
        """
        Compute the Toomre Q value for the disk.
        """
        assert hasattr(self, 'sigma'), "Error: sigma not set"
        assert hasattr(self, 'cs'), "Error: sound speed not set"
        cs     = self.cs
        omk    = self.omk
        sigma  = self.sigma
        self.qtoomre = cs * omk / (nc.pi * nc.GG * sigma)

    def compute_hsurf(self):
        """
        Given the irradiation angle self.flang and the total vertical optical
        depth according to the opacity model self.meanopacitymodel, the vertical
        height of the disk is computed. The opacity carrier (presumably the
        dust) is assumed to be vertically well-mixed with the gas. This
        can be computed from the pressure scale height, of the gas self.hp,
        and assuming that the gas (and dust) behave as a perfect Gaussian
        in vertical direction. It computes the height above the disk
        where the stellar radiation is absorbed and re-emitted, i.e. the
        Chiang & Goldreich (1997) warm surface layer.

        We follow Dullemond \& Dominik (2001), solving their equation A9.

        NOTE: This method uses the Planck mean opacity array
              self.mean_opacity_planck[:]. If you are not sure if this
              is still up-to-date with the current surface density and
              temperature, then better call self.compute_mean_opacity()

        EXAMPLE:
        Here is an example how this can be used without explicitly referring
        to a dust component (i.e. without using self.add_dust). Instead the
        vertical optical depth is computed from sigma, a dust-to-gas ratio
        0.01 and a dust opacity value kappa (which we simply take 1e3 for
        now).

        from disklab.diskradial import *
        from disklab.natconst import *
        import matplotlib.pyplot as plt
        opac=['supersimple',{'dusttogas':0.01,'kappadust':1e3}]
        d=DiskRadialModel(mdisk=0.01*MS,meanopacitymodel=opac)
        d.compute_hsurf()

        This produces d.hs, which is the surface height. You can most
        easily inspect the result by plotting d.hs/d.hp, which is the
        ratio of surface to pressure scale height.
        """
        assert hasattr(self, 'mean_opacity_planck'), 'ERROR: Cannot compute surface height without mean_opacity_planck.'
        assert hasattr(self, 'flang'), 'ERROR: Cannot compute surface height without self.flang.'

        # Interal function

        def fsurf(chi, rhs):
            from math import erf
            return 1. - erf(chi / 1.4142135623730951) - rhs
        #
        # Check
        #
        assert hasattr(self, 'mean_opacity_planck'), 'You must run compute_mean_opacity() first, before calling compute_hsurf()'
        #
        # Now loop over radius
        #
        nr      = len(self.r)
        self.hs = np.zeros(nr)
        flang   = np.zeros(nr) + self.flang
        tau     = self.sigma * self.mean_opacity_planck
        for ir in range(nr):
            if tau[ir] / flang[ir] > 6.:
                rhs         = 2 * flang[ir] / tau[ir]
                self.hs[ir] = self.hp[ir] * optimize.brentq(fsurf, 0.1, 10., args=(rhs))

    def compute_flareindex(self, pairwise=True, flmin=1e-2, expolin=True):
        """
        If we have computed self.hs (see function self.compute_hsurf()) at all radii,
        we can self-consistently compute the flaring index self.flidx, instead of
        simply having to set it to some value. From self.flidx the flaring angle
        can then be computed. One can then iterate to get the fully self-consistent
        solution. However, this is numerically unstable. This problem is solved by
        Chiang et al. ApJ 547:1077-1089 (2001), see their appendix. The trick is
        to do this pairwise.

        ARGUMENTS:
        pairwise   = If True, then use Chiang's pairwise method. Else continuous
                     (but in both cases shifted 1.5 cell, unfortunately)
        flmin      = The lower limit to the flaring index
        expolin    = If True, then copy the flidx[2] to flidx[0] and flidx[1]

        EXAMPLE:
        Same simplifications as above with compute_hsurf().

        from disklab.diskradial import *
        from disklab.natconst import *
        import matplotlib.pyplot as plt
        opac=['supersimple',{'dusttogas':0.01,'kappadust':1e3}]
        d=DiskRadialModel(mdisk=0.01*MS,meanopacitymodel=opac)
        d.compute_hsurf()
        d.compute_flareindex()
        flpair=d.flidx.copy()
        d.compute_flareindex(pairwise=False)
        flcont=d.flidx.copy()
        from plot import *
        plot(d.r/au,flpair,xlog=True)
        plot(d.r/au,flcont,oplot=True)

        This shows the pairwise flaring index compared to the continuous one.
        NOTE: The flaring index is always calculated 1.5 grid points to the right.
        This is because of the above mentioned Chiang et al. pairwise method to
        ensure stability.
        """
        nr         = len(self.r)
        if not hasattr(self, 'flidx'):
            self.flidx = np.zeros(nr)
        # flang = np.zeros(nr) + self.flang
        if pairwise:
            cntreset = 1    # Pairwise following Chiang et al. 2001
            flcnt    = 2
        else:
            cntreset = -1   # Continuous
            flcnt    = 0
        for ir in range(2, nr):
            if flcnt > cntreset:
                hr2 = self.hs[ir - 2] / self.r[ir - 2]
                hr1 = self.hs[ir - 1] / self.r[ir - 1]
                if hr1 > 0.0 and hr2 > 0.0:
                    self.flidx[ir] = (
                        self.r[ir - 2] + self.r[ir - 1]) * (hr2 - hr1) / ((
                            self.r[ir - 2] - self.r[ir - 1]) * (hr2 + hr1))
                else:
                    if ir > 0:
                        self.flidx[ir] = self.flidx[ir - 1]
                if self.flidx[ir] < flmin:
                    self.flidx[ir] = flmin
                flcnt = 0
            else:
                if ir > 0:
                    self.flidx[ir] = self.flidx[ir - 1]
            flcnt += 1
        if expolin:
            self.flidx[0:2] = self.flidx[2]

    def compute_flareangle_from_flareindex(self, inclrstar=True):
        """
        Compute the flaring angle self.flang from the flaring index
        self.flidx (computed, e.g., by self.compute_flareindex()).

        ARGUMENT:
        inclrstar   = If true, then add the 0.4*R_star/r term from
                      Chiang & Goldreich (1997), their eq. 5.
        """
        self.flang = self.flidx * self.hs / self.r
        if inclrstar:
            self.flang += 0.4 * self.rstar / self.r

    def iterate_flaringangle(self, inclrstar=True, iterhp=True, itermax=20,
                             convcrit=1e-2, keeptvisc=False):
        """
        Self-consistently compute the surface height of the disk, and
        the flaring (irradiation) angle. This is done by iterating
        the computation of the flaring index and the computation of
        the flaring angle.

        ARGUMENT:
        inclrstar   = If true, then add the 0.4*R_star/r term from
                      Chiang & Goldreich (1997), their eq. 5.
        iterhp      = If True, then also recalculate the midplane temperature,
                      the midplane sound speed and the pressure scale height
                      at each iteration.
        itermax     = Maximum number of iterations before error message.
        convcrit    = Relative change in the flaring angle as convergence
                      criterion.
        keeptvisc   = If true and if self.tvisc is present, then this is
                      included as a background temperature.

        EXAMPLE:
        from disklab.diskradial import *
        from disklab.natconst import *
        import matplotlib.pyplot as plt
        opac=['supersimple',{'dusttogas':0.01,'kappadust':1e3}]
        d=DiskRadialModel(mdisk=0.01*MS,meanopacitymodel=opac)
        plt.plot(d.r/au,d.flang+np.zeros(len(d.r)))
        iter=d.iterate_flaringangle(inclrstar=True,itermax=20,convcrit=1e-2)
        plt.plot(d.r/au,d.flang)
        plt.xscale('log')
        print('Nr of iterations = {}'.format(iter))

        One can see that this procedure typically converges very fast, at
        least for sufficiently optically thick disks.
        """
        if iterhp:
            self.compute_disktmid(keeptvisc=keeptvisc)
            self.compute_cs_and_hp()
            self.compute_mean_opacity()
        for iter in range(itermax):
            flang = self.flang
            self.compute_hsurf()
            self.compute_flareindex()
            self.compute_flareangle_from_flareindex(inclrstar=inclrstar)
            if iterhp:
                self.compute_disktmid(keeptvisc=keeptvisc)
                self.compute_cs_and_hp()
                self.compute_mean_opacity()
            change = np.abs((self.flang - flang) / (self.flang + flang))
            if change.max() < convcrit:
                self.compute_rhomid_from_sigma()
                return iter
        raise StopIteration('Flaring angle did not converge')

    def add_planet_gap(self, apl, model, mpl=None, depth=None, width=None,
                       insigma=True, innu=False, smooth=None, log=False,
                       **kwargs):
        """
        Once you have an overall disk model set up, you can insert a planetary
        gap, according to some analytic recipe.

        ARGUMENTS:
        apl     = Semi-major axis of the planet [cm]
        mpl     = Mass of the planet [g] (for self-consistent models)
        depth   = Relative depth of the gap (for dummy models)
        width   = Relative width of the gap (for dummy models)
        model   = Model to be used:
                = 'gauss':  use a simple gaussian model, with `width` as standard
                  deviation and depth as factor in the exponential.
                = 'duffell': use the Duffell (2015) ApJL 807, 11 model.
                  Additional keywords that can be used for this model:
                    - smooth: If set to e.g. 2, then the Duffell model is
                      smoothed, so that it does not have this sudden flat bottom.
                    - log: If set to True, then the Duffell model is slightly
                      adapted to write r/a-1 as ln(r/a)
                = 'crida': use the Crida et al. (2006) Icarus 181, 587 model.
                    options can be given as additional keywords:
                    - n_rh out to how many hill radii to carry out the integral
                      default: 12
                    - r_min: similar to graviational smoothing length to avoid
                      singularity in integral close to the planet. In units of
                      hill radii
                      default: 0.1
                    - smooth: number of hill radii over which the profile
                      is smoothly transitioned back to the original profile.
                      This is not used by default.
                = 'crida-fung': same as 'crida', but the gap depth is limited
                   according to Fung et al. 2014, ApJ 782, 88
                = 'kanagawa': use model of Kanagawa et al. (2017) PASJ 69:6, 97
                    - smooth: number of hill radii over which the profile
                      is smoothly transitioned back to the original profile.
                      This is not used by default.
        insigma = If True, then the gap is inserted into the surface density of
                  the gas and dust.
        innu    = If True, then the gap is implemented by increasing the viscosity
                  at the gap location, so that the gap remains even if the
                  disk is viscously evolved.

        WARNING: If you use innu=True, then the viscosity in the gap goes up
        by a lot. This is not a real (physical) viscosity increase! In reality
        it is a torque by the planet. But in the disk model it is, for now, included
        as if it is a viscosity increase. If you also model the dust drift and
        mixing, then this also increases the turbulent mixing, which is unphysical.
        The solution is to set self.alphamix = self.alpha before adding the first
        planet.
        """
        #
        # Pick a model
        #
        if model.lower() == 'gauss':
            #
            # Simple Gaussian gap model
            #
            assert depth is not None, "Error: Gauss gap model requires depth"
            assert width is not None, "Error: Gauss gap model requires width"
            #
            # Make the gauss factor
            #
            factor    = np.exp(-depth * np.exp(-0.5 * (self.r - apl)**2 / width**2))
        elif model.lower() == 'duffell':
            #
            # Duffell (2015) ApJL 807, 11.
            #
            assert mpl is not None, "Error: Duffell model requires planet mass"
            #
            # Compute the gap depth according to his Eqs.~(9, 10)
            #
            nr        = len(self.r)
            q         = mpl / self.mstar
            hp        = np.interp(apl, self.r, self.hp)
            mach      = apl / hp
            f0        = 0.45        # Taken from text below Eq. 10
            alpha     = np.interp(apl, self.r, (np.zeros(nr) + self.alpha))
            kconst    = q**2 * mach**5 / alpha
            gapdepth  = 1. / (1. + f0 * kconst / (3 * nc.pi))
            #
            # check the gap opening mass
            # Goodman & Rafikov 2001, Eq. 19
            #
            m_gap = 2 / 3 * self.mstar * (hp / apl)**3
            if mpl > m_gap:
                warnings.warn('Planet mass is larger than gap opening mass ({:.2g} M_jup) - the Duffell model should not be used'.format(m_gap / nc.Mju))
            #
            # Compute tsh (tau_shock) from Eq. 14
            #
            tsh       = 1.89 + 0.53 / (q * mach**3)
            #
            # Compute tau from Eq. 15
            # NOTE: This is approximate. At some point I have to build in Eq. 16
            #
            if log:
                #
                # Adapted formula with r/a-1 ---> ln(r/a)
                #
                tau   = (2.**0.75 / 5.) * np.abs(1.5 * mach * np.log(self.r / apl))**2.5
            else:
                #
                # Original formula
                #
                tau   = (2.**0.75 / 5.) * np.abs(1.5 * mach * (self.r / apl - 1.))**2.5
            #
            # Make the function f from Eq. 13
            #
            if smooth is None:
                #
                # Original formula by Duffell
                #
                f         = f0 * np.sqrt(tsh / tau)
                f[tau < tsh] = f0
            else:
                #
                # Adapted formula that is smoother at the bottom
                # (not part of original Duffell paper)
                #
                tauratio  = ((tau / tsh)**smooth + 1.)**(-1. / smooth)
                f         = f0 * np.sqrt(tauratio)
            #
            # Compute the multiplicative factor in Eq. 12
            #
            factor    = 1. - np.sqrt(apl / self.r) * (f * kconst / (3 * nc.pi)) / (1. + f0 * kconst / (3 * nc.pi))
            #
            # Make sure that this factor never becomes smaller than, say, 0.8*gapdepth
            # (the 0.8 is just an arbitrary number) to ensure that this factor does
            # not become negative.
            #
            factor[factor < 0.8 * gapdepth] = 0.8 * gapdepth
        elif model.lower().startswith('crida'):

            assert mpl is not None, "Error: Crida model requires planet mass"
            n_rh = kwargs.pop('n_rh', 12)
            r_min = kwargs.pop('r_min', 0.1)

            def a_pp(x):
                "Crida et al. 2006, Eq. 13"
                return 1. / 8. * np.abs(x)**-1.2 + 200 * np.abs(x)**-10

            def t_g(x, delta, r_p, q, omega_p):
                """
                Returns the gravitational torque according to Crida+06, Eq. 11

                Arguments:
                ----------

                x : float
                    radial position where to evaluate t_g

                delta : float
                    distance to planet

                r_p : float
                    position of planet in same units as d.r

                q : float
                    planet-to-star mass ratio [-]

                omega_p : float
                    orbital frequency at planet position [s**-1]
                """
                return 0.35 * q**2 * r_p**5 * omega_p**2 * x * delta**-4 * np.sign(delta)

            def crida_RHS(x, r_p, q):
                """
                (r_H/Sigma) * (dSigma / dr), according to Eq. 14 in Crida et al. 2006.

                Arguments:
                ----------

                x : float
                    radial position where to evaluate the integrand

                r_p : float
                    position of planet

                q : float
                    planet-to-star mass ratio [-]

                """
                omega_p = np.sqrt(nc.GG * self.mstar / r_p**3)
                r_H = r_p * (q / 3.)**(1. / 3.)
                delta = x - r_p

                # XXXXX EXPERIMENTAL: AVOID SINGULARITY XXXXXX
                if delta >= 0:
                    delta = max(delta, r_min * r_H)
                if delta < 0:
                    delta = -max(np.abs(delta), r_min * r_H)
                # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

                if np.abs(delta) < 2 * r_H:
                    return 0

                cs = np.interp(x, self.r, self.cs)
                hp = np.interp(x, self.r, self.hp)
                alpha = np.interp(x, self.r, self.alpha * np.ones_like(self.r))

                return (t_g(x, delta, r_p, q, omega_p) - 0.75 * alpha * cs**2) /  \
                    ((hp / x)**2 * x * r_p * omega_p**2 * a_pp(delta / r_H) + 1.5 * alpha * cs**2 * x / r_H)

            q = mpl / self.mstar
            r_H = apl * (q / 3.)**(1. / 3.)

            # make a grid around the planet position
            mask = (self.r > apl - n_rh * r_H) & (self.r <= apl + n_rh * r_H)
            r_sub = self.r[mask]

            # integrate between all the cells, then add things up to get a
            # cumulative integral

            # integral_parts = np.array([quad(crida_RHS, r_sub[i], r_sub[i + 1], args=(apl, q))[0] for i in range(len(r_sub) - 1)])
            # integral = np.hstack((0, np.cumsum(integral_parts)))
            integral_parts = np.array([quad(crida_RHS, r_sub[i], r_sub[i - 1], args=(apl, q))[0] for i in range(len(r_sub) - 1, 0, -1)])
            integral = np.hstack((0, np.cumsum(integral_parts)))[::-1]

            # calculate the multiplicative factor
            factor = np.ones_like(self.r)
            factor[mask] = np.exp(integral / r_H)

            if model.lower() == 'crida-fung':
                mask2 = (self.r > apl - 2 * r_H) & (self.r <= apl + 2 * r_H)

                hp = self.hp[mask2]
                alpha = (self.alpha * np.ones_like(self.r))[mask2]
                f_min = np.ones_like(self.r[mask2])

                if 1e-4 <= q <= 5e-3:
                    # Eq. 12 in Fung+14
                    f_min = 1.4e-1 * (q / 1e-3)**-2.16 * (alpha / 1e-2)**1.41 * (hp / (self.r[mask2] * 0.05))**6.61
                elif 5e-3 < q:
                    # Eq. 14 in Fung+14
                    f_min = 4.7e-3 * (q / 5e-3)**-1.00 * (alpha / 1e-2)**1.26 * (hp / (self.r[mask2] * 0.05))**6.12

                # factor[mask2] = np.maximum(f_min, factor[mask2])
                factor[mask2] = f_min

        elif model.lower().startswith('kanagawa'):
            hp = np.interp(apl, self.r, self.hp)
            alpha = np.interp(apl, self.r, self.alpha * np.ones_like(self.r))

            K = (mpl / self.mstar)**2 * (hp / apl)**-5 / alpha
            Kp = (mpl / self.mstar)**2 * (hp / apl)**-3 / alpha

            factor_min = 1. / (1. + 0.04 * K)  # Eq. 11

            delta_R_1 = (factor_min / 4. + 0.08) * Kp**0.25 * apl  # Eq. 8
            delta_R_2 = 0.33 * Kp**0.25 * apl  # Eq. 9

            factor_gap = 4 * Kp**-0.25 * np.abs(self.r - apl) / apl - 0.32  # Eq. 7

            # Eqn. 6

            factor = np.ones_like(self.r)

            mask1 = np.abs(self.r - apl) < delta_R_1
            mask2 = (delta_R_1 <= np.abs(self.r - apl)) & (np.abs(self.r - apl) <= delta_R_2)

            factor[mask1] = factor_min
            factor[mask2] = factor_gap[mask2]
        else:
            raise ValueError('Model for gap not known.')

        #  smoothly go back to original profile outside of planet
        if model.lower() in ['crida', 'kanagawa'] and smooth is not None:
            r_H = apl * (mpl / (self.mstar * 3.))**(1. / 3.)
            drsmooth = smooth * r_H
            factor = np.exp(np.log(factor) * np.exp(-0.5 * (self.r - apl)**4 / drsmooth**4))
        #
        # Now implement this into the sigma and/or viscosity
        #
        if insigma:
            self.sigma   *= factor
            self.rhomid  *= factor
            if hasattr(self, 'dust'):
                for d in self.dust:
                    d.sigma *= factor
                    d.compute_hdust_and_rhodust()
        if innu:
            if hasattr(self, 'nu'):
                self.nu  *= 1. / factor
            if hasattr(self, 'alpha'):
                self.alpha *= 1. / factor

    def compute_radial_derivative(self, q, interp=True, dbllog=False):
        """
        It is often needed to compute the derivative of a quantity q
        with respect to r. This is the standardized routine for it.
        If dbllog is True, then we take the double-logarithmic derivative
        dln(q)/dln(r). If not, then the normal derivative dq/dr.
        If interp=True then we re-map the results back to the actual
        grid. If not, the results are valid between the grid points.
        """
        if dbllog:
            lq     = np.log(q)
            lr     = np.log(self.r)
            der    = (lq[1:] - lq[:-1]) / (lr[1:] - lr[:-1])
        else:
            q      = q
            r      = self.r
            der    = (q[1:] - q[:-1]) / (r[1:] - r[:-1])
        if interp:
            derc   = 0.5 * (der[1:] + der[:-1])
            derl   = 2 * derc[0] - derc[1]
            derr   = 2 * derc[-1] - derc[-2]
            der    = np.hstack((derl, derc, derr))
        return der

    def compute_omega(self, vertint=False, interp=True):
        """
        Compute the actual value of the orbital frequency of the
        gas Omega. This should be very close to the Kepler value
        Omega_K, but not exactly. This is because of the radial
        pressure gradient. This is a small effect, but it can be
        important for dust drift as well as for the Rayleigh
        stability of gas rings.

        NOTE: This equation can be computed using the midplane density
        and pressure OR using the verically integrated density (=surface
        density) and vertically integrated pressure. In Li et al. they
        use the vertically intergrated version. You can switch this
        with vertint=False/True.

        With interp=True the results (which are initially calculated
        between the grid points) are interpolated onto the gridpoints
        again. This means, however, that the first and last values are
        not accurate.

        Computes: dvphi = v_phi-v_K, omega, dlnpdlnr = dlog(P)/dlog(r),
        vphi = v_phi, lphi = v_phi*r.
        """
        r     = self.r
        ri    = 0.5 * (r[1:] + r[:-1])
        omki  = (nc.GG * self.mstar / ri**3)**0.5
        vki   = omki * ri
        csi   = 0.5 * (self.cs[1:] + self.cs[:-1])
        dlnr  = (np.log(r[1:]) - np.log(r[:-1]))
        if vertint:
            pr   = self.sigma * self.cs**2
        else:
            pr   = self.rhomid * self.cs**2
        dlnpr  = (np.log(pr[1:]) - np.log(pr[:-1]))
        dlnpdlnr = dlnpr / dlnr
        #
        # In Lynden-Bell & Pringle type solutions this could become
        # too negatively large (due to the exponential cut-off).
        # So we will limit this value.
        #
        dum   = 1.001 * np.abs(dlnpdlnr) * (csi / vki)**2
        dum[dum < 1.0] = 1.0
        dlnpdlnr = dlnpdlnr / dum
        #
        # Now calculate the actual orbital angular frequency, the
        # specific angular momentum, the angular velocity and the
        # velocity difference to Kepler.
        #
        omega = omki * (1.0 + (csi / vki)**2 * dlnpdlnr)**0.5
        lphi  = omega * ri**2
        vphi  = omega * ri
        dvphi = (omega - omki) * ri
        #
        # If the caller wants the results interpolated at the cell
        # centers, then do so.
        #
        if interp:
            omega    = 0.5 * (omega[1:] / omki[1:] + omega[:-1] / omki[:-1]) * self.omk[1:-1]
            ominn    = (2 * (omega[0] / self.omk[1]) - omega[1] / self.omk[2]) * self.omk[0]
            omout    = (2 * (omega[-1] / self.omk[-2]) - omega[-2] / self.omk[-3]) * self.omk[-1]
            omega    = np.hstack((ominn, omega, omout))
            dlnpdlnr = 0.5 * (dlnpdlnr[1:] + dlnpdlnr[:-1])
            dlnpdlnr = np.hstack((dlnpdlnr[0], dlnpdlnr, dlnpdlnr[-1]))
            lphi     = omega * self.r**2
            vphi     = omega * self.r
            dvphi    = (omega - self.omk) * self.r
        self.lphi     = lphi
        self.vphi     = vphi
        self.dvphi    = dvphi
        self.omega    = omega
        self.dlnpdlnr = dlnpdlnr

    def compute_solberg_hoiland(self, vertint=False, interp=True, gam=None):
        """
        Compute the value of the Solberg Hoiland criterion number:
          SH = kappa^2 + N^2
        If SH > 0 then the disk is stable. If SH < 0 then the disk is
        Rayleigh unstable. We follow Li, Finn, Lovelace & Colgate (2000)
        ApJ 533, 1023, Equation 22.
        NOTE: This equation can be computed using the midplane density
        and pressure OR using the verically integrated density (=surface
        density) and vertically integrated pressure. In Li et al. they
        use the vertically intergrated version. Note however, that the
        test cases shown in Li et al. 2000 appear to have a difference
        with our model: In our model, when there is a global pressure
        gradient, the Omega < Omega_K even far from the bump or jump.
        In their plots, however, outside the bump or jump Omega==Omega_K.
        Their equations for the kappa^2 and N^2 are, however, consistent
        with another paper: Yang & Menou (2010), although the latter is
        more general since it includes also the vertical gradients, which
        we do not include.

        EXAMPLE: Disk with a bump (our model)

        from disklab.diskradial import *
        from disklab.natconst import *
        import matplotlib.pyplot as plt
        a=DiskRadialModel(mdot=1e-8*MS/year,rin=10*au,rout=30*au,nr=1000)
        rbump=20.*au                         # Radial location of bump
        abump=1.0                            # Amplitude of bump (relative)
        hpbump=np.interp(rbump,a.r,a.hp)     # Pressure scale height
        wbump=0.5*hpbump                     # Width (stand dev) of Gaussian bump
        fact=1+abump*np.exp(-0.5*((a.r-rbump)/wbump)**2)
        a.sigma *= fact
        a.rhomid *= fact
        a.compute_solberg_hoiland(vertint=True)  # Compute SH with Sigma and P=Sigma*c_s^2
        SHsqdimless = a.SHsq / a.omk**2      # SH freq in units of Kepler freq
        plt.plot(a.r/au,SHsqdimless)
        plt.show()
        """
        #
        # Compute specific angular momentum at grid points
        #
        self.compute_omega(vertint=vertint, interp=True)
        #
        # Compute d(l^2)/dr / r^3 between the grid points, and
        # accordingly the kappa^2 as given above Eq. 22 of Li et al.
        #
        r     = self.r
        ri    = 0.5 * (r[1:] + r[:-1])
        lsq   = self.lphi**2
        kapsq = self.compute_radial_derivative(lsq, interp=False) / ri**3
        #
        # Next compute the Brunt-Vaisala N^2 between the grid points
        # Eq. 22 of Li et al. (equivalent to Eq. 13 of Yang & Menou 2010)
        #
        gamt  = 5. / 3.    # Gas polytropic index
        if vertint:
            if gam is None:
                gam  = (3 * gamt - 1.) / (gamt + 1.)   # Li & Lovelace top page 1026
            rho  = self.sigma
            pr   = self.sigma * self.cs**2
        else:
            if gam is None:
                gam  = gamt
            rho  = self.rhomid
            pr   = self.rhomid * self.cs**2
        rhoi   = 0.5 * (rho[1:] + rho[:-1])
        pri    = 0.5 * (pr[1:] + pr[:-1])
        dprdr  = self.compute_radial_derivative(pr, interp=False)
        drhodr = self.compute_radial_derivative(rho, interp=False)
        Nsq    = (1 / rhoi) * dprdr * (drhodr / rhoi - dprdr / pri / gam)
        #
        # If the user wants the quantities at the grid cells (instead
        # of in between the grid cells), we have to interpolate
        #
        if interp:
            omksq  = nc.GG * self.mstar / self.r**3
            omksqi = nc.GG * self.mstar / ri**3
            kapsqi = 0.5 * (kapsq[1:] / omksqi[1:] + kapsq[:-1] / omksqi[:-1])
            kapsq  = np.hstack((kapsqi[0], kapsqi, kapsqi[-1])) * omksq
            Nsqi   = 0.5 * (Nsq[1:] / omksqi[1:] + Nsq[:-1] / omksqi[:-1])
            Nsq    = np.hstack((Nsqi[0], Nsqi, Nsqi[-1])) * omksq
        else:
            self.compute_omega(vertint=vertint, interp=False)
        #
        # Store the results
        #
        self.kapsq = kapsq
        self.Nsq   = Nsq
        self.SHsq  = kapsq + Nsq  # Where this is >=0 the disk is stable

    def add_dust(self, St=None, agrain=None, dtg=0.01, xigrain=3.0):
        """
        This routine adds a dust component to the model. If St (Stokes number)
        is given, it tries to derive the corresponding grain size. But that
        might (for large grains) depend on the relative velocity of the grain
        wrt the gas, which cannot be known at this stage. A guess of dv=1e3
        (10 m/s) is taken. If, however, agrain (grain radius in cm) is given,
        then everything is straightforward.

        The routine also computes the dust vertical scale height according to
        Eq. 51 of Birnstiel et al. (2010) A&A 513, 79. To compute the
        dynamic properties of the dust, we use the grainmodel.py module.

        Returns: Adds a new dust component to the list self.dust
        """
        # nr = len(self.r)
        assert St is not None or agrain is not None, "Must give either agrain or St."
        assert St is None or agrain is None, "Cannot give St and agrain simultaneously."
        dust = DiskRadialComponent(self, agrain=agrain, xigrain=xigrain, St=St, sigma=self.sigma * dtg)
        if not hasattr(self, 'dust'):
            self.dust = []
        self.dust.append(dust)

    def bplanck(self, freq, temp):
        """
        This function computes the Planck function

                       2 h nu^3 / c^2
           B_nu(T)  = ------------------    [ erg / cm^2 s ster Hz ]
                      exp(h nu / kT) - 1

        Arguments:
         freq  [Hz]            = Frequency in Herz (can be array)
         temp  [K]             = Temperature in Kelvin (can be array)
        """
        const1  = nc.hh / nc.kk
        const2  = 2 * nc.hh / nc.cc**2
        const3  = 2 * nc.kk / nc.cc**2
        x       = const1 * freq / (temp + 1e-99)
        if np.isscalar(x):
            if x > 500.:
                x = 500.
        else:
            x[np.where(x > 500.)] = 500.
        bpl     = const2 * (freq**3) / ((np.exp(x) - 1.e0) + 1e-99)
        bplrj   = const3 * (freq**2) * temp
        if np.isscalar(x):
            if x < 1.e-3:
                bpl = bplrj
        else:
            ii      = x < 1.e-3
            bpl[ii] = bplrj[ii]
        return bpl

    def tbright_equation(self, temp, freq, intensity):
        eq = self.bplanck(freq, temp) - intensity
        return eq

    def solve_tbright(self, freq, intensity, linear=False):
        """
        Compute the brightness temperature of a given intensity
        at a given frequency. The temperature is by definition such that
        B_nu(T) = intensity. If, however, you set linear=True, then
        it will do the usual linear (Rayleigh-Jeans) formula for the
        brightness temperature.

        ARGUMENTS:
          freq           Frequency in Hz
          intensity      Intensity in erg/cm^2/s/Hz/ster
          linear         If linear, then use Rayleigh-Jeans equation
                         instead of full Planck function (default=False)

        RETURNS:
          The brightness temperature in Kelvin
        """
        if not linear:
            tlow    = 1e-1
            thigh   = 1e4
            eqlow   = self.tbright_equation(tlow, freq, intensity)
            eqhigh  = self.tbright_equation(thigh, freq, intensity)
            if eqlow > 0.:
                tbright = tlow
            elif eqhigh < 0.:
                tbright = thigh
            else:
                tbright = optimize.brentq(self.tbright_equation, tlow, thigh, args=(freq, intensity))
        else:
            const3  = 2 * nc.kk / nc.cc**2
            tbright = intensity / (const3 * (freq**2))
        return tbright

    def compute_onezone_intensity(self, lam, dust=None, inclbg=True, inclscat=True, inclang=0.):
        """
        Compute the intensity of the disk seen face-on, assuming no warm surface
        layer, only the midplane.

        ARGUMENTS:
          lam              Wavelength in [cm]
          dust             List of dust components to include in the
                           radiative transfer calculation. Default: self.dust
          inclbg           If True, then include the background Planck emission
                           with temperature self.tbg.
          inclscat         If True, then include the effect of scattering.
          inclang          The inclination angle (default=0=face-on) in [radian]

        RETURNS:
          self.intensity   The intensity as a function of r in erg/cm^2/s/Hz/ster
          self.flux_at_oneparsec   The flux in erg/cm^2/s/Hz seen at 1 pc distance

        EXAMPLE:

        from disklab.diskradial import *
        from disklab.grainmodel import *
        from disklab.natconst import *
        import matplotlib.pyplot as plt
        a=DiskRadialModel(mdot=1e-8*MS/year,alpha=1e-3)
        a.add_dust(agrain=1e-1)     # a = 1 mm dust
        lam = np.array([1e-1,1e-2]) # lambda = 1.0, 0.1 mm
        a.dust[0].grain.compute_simple_opacity(lam,tabulatemean=False)
        plt.figure()
        a.compute_onezone_intensity(lam)
        plt.plot(a.r/au,a.intensity[0,:])
        plt.plot(a.r/au,a.intensity[1,:])
        plt.xlabel(r'$r\; \mathrm{[au]}$')
        plt.ylabel(r'$I_\nu(r)\; \mathrm{[erg/cm^2/s/Hz/ster]}$')
        plt.xscale('log')
        plt.yscale('log')
        plt.show()
        """
        if dust is None:
            dust = self.dust
        if type(dust) != list and type(dust) != np.ndarray:
            dust = [dust]
        if type(lam) != np.ndarray:
            if type(lam) != list:
                lam = [lam]
            lam = np.array(lam)
        self.lam  = lam
        self.freq = nc.cc / lam
        lammic    = lam * 1e4
        nf        = len(lam)
        nr        = len(self.r)
        tauabs    = np.zeros((nf, nr))
        tauscaeff = np.zeros((nf, nr))
        tau       = np.zeros((nf, nr))
        mu        = np.cos(inclang)
        #
        # Now add all the tau_abs of the dust species together (for each r and each frequency)
        #
        for d in dust:
            assert hasattr(d, 'grain'), "Can only do radiative transfer if the dust components have a dust.grain attribute (see grainmodel.py), which contains the grain opacity information."
            g = d.grain
            assert hasattr(g, 'opac_lammic') and hasattr(g, 'opac_kabs'), "Can only do radiative transfer if the dust components have opacity information (i.e. dust.grain.opac_lammic and dust.grain.opac_kabs)"
            kapabs    = np.interp(lammic, g.opac_lammic, g.opac_kabs)
            gsca      = np.zeros(nf)
            if inclscat:
                kapsca    = np.interp(lammic, g.opac_lammic, g.opac_ksca)
                if hasattr(g,'opac_gsca'): gsca = np.interp(lammic, g.opac_lammic, g.opac_gsca)
            else:
                kapsca    = np.zeros(nf)
            kapscaeff = kapsca * (1.0 - gsca)  # Ishimaru 1978
            for inu in range(nf):
                tauabs[inu,:]    += d.sigma * kapabs[inu]
                tauscaeff[inu,:] += d.sigma * kapscaeff[inu]
        #
        # Now use the simple one-zone radiative transfer equation to compute the intensity (for each r and each frequency)
        #
        self.intensity = np.zeros((nf, nr))
        self.tau = np.zeros((nf, nr))
        self.tauobs = np.zeros((nf, nr))
        for inu in range(nf):
            if inclbg:
                bplbg = self.bplanck(self.freq[inu], self.tbg)
            else:
                bplbg = 0.
            bpl     = self.bplanck(self.freq[inu], self.tmid)
            if not inclscat:
                #
                # Simple radiative transfer without scattering
                #
                tau     = tauabs[inu,:] + tauscaeff[inu,:]
                tauobs  = tau/mu
                xp      = np.exp(-tauobs)
                xpt     = 1. - xp
                ii      = tauobs < 1e-6
                xpt[ii] = tauobs[ii]
                self.intensity[inu,:] = xp * bplbg + xpt * bpl
                self.tau[inu,:] = tau.copy()
                self.tauobs[inu,:] = tauobs.copy()
            else:
                #
                # A simplified treatment of scattering, according to
                # Birnstiel, Dullemond, et al. (2018) ApJ 869, L45
                # Section 5.
                #
                def scatsolution_eddbarbier(tauabs,tauscaeff,bpl,bplbg,mu):
                    tau      = tauabs + tauscaeff
                    epseff   = tauabs / tau                         # Eq.(13)
                    sepseff  = np.sqrt(epseff)
                    sepseff3 = np.sqrt(3*epseff)
                    xp3      = np.exp(-sepseff3*tau)
                    b        = 1./((1.-sepseff)*xp3+(1.+sepseff))   # Eq.(16)
                    taueddb  = (2./3.)*mu + np.zeros_like(epseff)   # This taueddb is in fact 0.5*Delta\tau-tau from the paper
                    mask     = (2./3.)*mu>tau
                    taueddb[mask] = tau[mask]                       # Limit taueddb to at most tau
                    expl     = np.exp(-sepseff3*taueddb)            # First term in Brackets in Eq.(15)
                    expr     = np.exp(sepseff3*(taueddb-tau))       # Second term in Brackets in Eq.(15)
                    xpp      = b*(expl+expr)
                    meanint  = xpp*bplbg + (1.-xpp)*bpl             # Eq.(15) plus background radiation
                    source   = epseff*bpl + (1.-epseff)*meanint     # Eq.(19)
                    tauobs   = tau / mu
                    xp       = np.exp(-tauobs)
                    xpt      = 1. - xp
                    ii       = tauobs < 1e-6
                    xpt[ii]  = tauobs[ii]
                    return xp * bplbg + xpt * source, tau.copy(), tauobs.copy()   # Eq.(18) plus background radiation, tau, tauobs
                self.intensity[inu,:], self.tau[inu,:], self.tauobs[inu,:] = \
                    scatsolution_eddbarbier(tauabs[inu,:],tauscaeff[inu,:],bpl,bplbg,mu)
                
        #
        # Finally compute the fluxes as measured face-on at 1 parsec distance
        #
        ds  = nc.pi * (self.r[1:]**2 - self.r[:-1]**2)
        iav = 0.5 * (self.intensity[:, 1:] + self.intensity[:, :-1])
        self.flux_at_oneparsec = np.zeros(nf)
        for inu in range(nf):
            self.flux_at_oneparsec[inu] = (iav[inu, :] * ds).sum() / (nc.pc**2)

    def compute_tbright_from_intensity(self, linear=False):
        """
        Once you have computed self.intensity (see above), then it
        might be useful to compute the brightness temperature from this.
        By default the brightness temperature is defined by fitting
        a Planck function to the intensity.
        """
        assert hasattr(self, 'intensity'), "No self.intensity available"
        assert hasattr(self, 'freq'), "No self.freq available"
        nr = len(self.r)
        nf = len(self.lam)
        self.tbright = np.zeros((nf, nr))
        for inu in range(nf):
            for ir in range(nr):
                self.tbright[inu, ir] = self.solve_tbright(self.freq[inu], self.intensity[inu, ir], linear=linear)

    def get_viscous_evolution_next_timestep(self, dt, alphamodel=True, sigma_innerbc=None, exttorque=None, mdot_outerbc=None):
        """
        Advance sigma one time step into the future using the viscous disk
        equation. The method is fully implicit, so there is no strict limit
        on the time step. But of course, the smaller the time step, the better
        accuracy one gets.

        The viscosity used is the self.nu array. It will be automatically
        computed from self.alpha, UNLESS alphamodel==False (in which case
        it will use the values of self.nu given).

        The solution method is by casing the standard viscous disk equation
        into the form of a standard diffusion equation by making the following
        definitions: y = 2*pi*r*sigma, x = r, g = sqrt(r)/nu, d = 3 * nu.
        We then use a standard diffusion equation solver solvediffonedee.py.

        The resulting sigma is not replacing the old one, but is returned as
        an argument. If you want to replace the old one, simply set self.sigma
        to the returned sigma.

        IMPORTANT:
        The default inner boundary condition is to set the gradient of
        Sigma*r zero. This works ok for simple temperature profiles. But
        for more complex ones, this leads to instabilities. You may then
        want to set sigma_innerbc=1e-3 or some other low value. One way
        to check if things go wrong is to compute the accretion rate mdot,
        and assure that mdot>0 at the inner edge:
          d.compute_mdot_at_interfaces()
          assert d.mdot[0]>=0., 'Problem: outward pointing flow at inner boundary!'

        With exttorque you can specify an external torque in units of
        cm^2/s^2 per gram. This gives an extra radial velocity of
        vr += 2*exttorque/v_k, where v_k is the Kepler velocity.

        EXAMPLE USAGE:

        from disklab.diskradial import *
        from disklab.natconst import *
        import matplotlib.pyplot as plt
        a=DiskRadialModel(mdisk=0.01*MS)
        a.sigma[a.r>1*AU]=1e-6         # Remove disk where r>1AU (just to make nice small disk)
        plt.plot(a.r/au,a.sigma)
        plt.xscale('log')
        plt.yscale('log')
        s=a.get_viscous_evolution_next_timestep(1e3*year)
        plt.plot(a.r/au,s)
        s=a.get_viscous_evolution_next_timestep(1e4*year)
        plt.plot(a.r/au,s)
        s=a.get_viscous_evolution_next_timestep(1e5*year)
        plt.plot(a.r/au,s)
        plt.ylim(ymin=1e-6)
        plt.show()

        IMPORTANT: Here the evolution is done in ONE time step. This is just an approximation,
        because the time-derivative is then not accurate. For a real solution you have to
        split this into multiple time steps.

        """
        #
        # If requested, compute nu from alpha
        #
        if alphamodel:
            self.compute_nu()
        #
        # Cast into diffusion equation form
        #
        x    = self.r
        y    = 2 * nc.pi * self.r * self.sigma
        g    = self.r**0.5 / self.nu
        d    = 3 * self.nu
        v    = np.zeros(len(x))
        #
        # Source term only if self.sigdot is given
        #
        if hasattr(self, 'sigdot'):
            s = 2 * nc.pi * self.r * self.sigdot
        else:
            s = np.zeros(len(x))    # For now no source term
        #
        # Set boundary conditions
        #
        if sigma_innerbc is None:
            bcl  = (1, 0, 0, 0)
        else:
            bcl  = (0, 1, 2 * nc.pi * self.r[0] * sigma_innerbc, 0)
        if mdot_outerbc is None:
            bcr  = (0, 1, 2 * nc.pi * self.r[-1] * self.sigmin, 0)
        # opedit: adding fixed external mdot boundary condition here:
        else:
            bcr  = (1, 0, mdot_outerbc/3./self.r[-1]**0.5, 1)
        #
        # Add additional torque, if requested
        #
        if exttorque is not None:
            vk   = self.omk * self.r
            v[:] = 2 * exttorque / vk
        #
        # Get the new value of y after one time step dt
        #
        y    = solvediffonedee(x, y, v, d, g, s, bcl, bcr, dt=dt, int=False)
        #
        # Obtain new sigma
        #
        sigma = y / (2 * nc.pi * self.r)
        #
        # Check
        #
        if sigma.min() < 0.0:
            print("Warning: negative surface density found.")
            if sigma_innerbc is None:
                print("Advice: Maybe this problem can be fixed by setting sigma_innerbc=1e-3 or some other small value.")
        #
        # Return
        #
        return sigma

    def compute_viscous_evolution_next_timestep(
            self, dt, alphamodel=True,
            vischeat=False, sigma_innerbc=None, mdot_outerbc=None):
        """
        Wrapper routine around get_viscous_evolution_next_timestep().
        Instead of returning the new sigma, it stores sigma and
        recomputes the midplane density and temperature, and the
        resulting new mean opacities.

        IMPORTANT:
        The default inner boundary condition is to set the gradient of
        Sigma*r zero. This works ok for simple temperature profiles. But
        for more complex ones, this leads to instabilities. You may then
        want to set sigma_innerbc=1e-3 or some other low value. One way
        to check if things go wrong is to compute the accretion rate mdot,
        and assure that mdot>0 at the inner edge:
          d.compute_mdot_at_interfaces()
          assert d.mdot[0]>=0., 'Problem: outward pointing flow at inner boundary!'
        """
        self.sigma = self.get_viscous_evolution_next_timestep(dt, alphamodel=alphamodel, sigma_innerbc=sigma_innerbc, mdot_outerbc=mdot_outerbc)
        self.compute_disktmid(vischeat=vischeat)
        self.compute_cs_and_hp()
        self.compute_rhomid_from_sigma()
        if hasattr(self, 'meanopacitymodel'):
            self.compute_mean_opacity()

    def compute_mdot_at_interfaces(self, alphamodel=True):
        """
        Compute the self-consistent gas accretion rate Mdot at the interfaces, using
        the same method as used for the get_viscous_evolution_next_timestep()
        method. In other words: the Mdot(r) computed here is self-consistent with
        the time-evolution. For the meaning of alphamodel, please look at the
        get_viscous_evolution_next_timestep().

        EXAMPLE USAGE:

        from disklab.diskradial import *
        from disklab.natconst import *
        import matplotlib.pyplot as plt
        a=DiskRadialModel(rout=1000*au)
        a.make_disk_from_lbp_alpha(1e-2*MS,1*au,1e-2,1e6*year)
        a.compute_mdot_at_interfaces()
        ri=0.5*(a.r[1:]+a.r[:-1])
        plt.plot(ri/au,a.mdot/(MS/year))
        plt.xscale('log')
        eta=(2.5-a.gam)/(2-a.gam)
        mdt=(eta-1)*(a.M0/a.tnu)*a.T**(-eta)
        plt.plot(a.r/au,mdt/(MS/year)+np.zeros(len(a.r)))
        plt.show()

        Where mdt is the Mdot value at the inner edge from the analytic model.
        See Lodato, Scardoni, Manara & Testi (2017) for these analytic formulae.
        As one can see: the mdot value at the inner edge is indeed the same as
        the mdt value, showing that the result appears to be correct.
        """
        #
        # If requested, compute nu from alpha
        #
        if alphamodel:
            self.compute_nu()
        #
        # Cast into diffusion equation form
        #
        x    = self.r
        y    = 2 * nc.pi * self.r * self.sigma
        g    = self.r**0.5 / self.nu
        d    = 3 * self.nu
        v    = np.zeros(len(x))
        #
        # Compute the mdot
        #
        self.mdot = -getfluxonedee(x, y, v, d, g, int=False)

    def compute_vr_at_interfaces(self, alphamodel=True, upwind=True):
        """
        Computes the radial gas velocity due to the viscous accretion at the interfaces.
        It does so by first calculating the gas Mdot at the interfaces, then dividing by
        2*pi*r*sigma. If upwind==True, then define the vr using the upwind scheme,
        otherwise using the average of 2*pi*r*sigma on both sides of the interface.

        EXAMPLE USAGE:

        from disklab.diskradial import *
        from disklab.natconst import *
        import matplotlib.pyplot as plt
        a=DiskRadialModel(rout=1000*au)
        a.make_disk_from_lbp_alpha(1e-2*MS,1*au,1e-2,1e6*year)
        a.compute_vr_at_interfaces()
        ri=0.5*(a.r[1:]+a.r[:-1])
        plt.plot(ri/au,a.vr)
        plt.xscale('log')
        vrr=-1.5*a.nu/a.r     # Analytic solution for steady-state
        plt.plot(a.r/au,vrr)
        plt.show()

        Here one sees that the analytic solution for the steady state indeed
        matches the actual solution for the inner disk, as expected.
        """
        #
        # Compute gas Mdot at the interfaces
        #
        self.compute_mdot_at_interfaces(alphamodel=alphamodel)
        #
        # Compute 2*pi*r*Sigma_g at the interfaces, using the correct averaging
        # or upwinding (note: Mdot has minus sign).
        #
        q    = np.zeros(len(self.r) - 1)
        if upwind:
            q += (self.mdot <= 0) * 2 * nc.pi * self.r[:-1] * self.sigma[:-1]
            q += (self.mdot > 0)  * 2 * nc.pi * self.r[1:] * self.sigma[1:]
        else:
            q = 0.5 * 2 * nc.pi * (self.r[:-1] * self.sigma[:-1] + self.r[1:] * self.sigma[1:])
        #
        # Now compute the radial velocity from the mdot
        #
        self.vr   = - self.mdot / (q + 1e-90)

    def compute_viscous_evolution_and_dust_drift_next_timestep(self, dt, alphamodel=True, extracond=None, updatestokes=True, vischeat=False, sigma_innerbc=None):
        """
        Wrapper routine combining both the gas viscous evolution and the
        dust radial drift.
        """
        self.compute_viscous_evolution_next_timestep(dt, alphamodel=alphamodel, vischeat=vischeat, sigma_innerbc=sigma_innerbc)
        for dust in self.dust:
            dust.compute_dust_radial_drift_next_timestep(dt, alphamodel=alphamodel, fixgas=False, extracond=extracond, updatestokes=updatestokes)

    def compute_shuulrich_infall(self, mcloud, cs, omega, time, idistr=1, get_tdisk=False):
        """
        Wrapper around the class ShuUlrichModel.
        """
        tcloud = self.mu*nc.mp*cs**2/nc.kk
        if not hasattr(self,'shuulrich'):
            self.shuulrich = ShuUlrichModel(mcloud=mcloud,tcloud=tcloud,omcloud=omega,idistr=idistr,rgrid=self.r)
        else:
            self.shuulrich.__init__(mcloud=mcloud,tcloud=tcloud,omcloud=omega,idistr=idistr,rgrid=self.r)
        if get_tdisk:
            self.infall_mdot      = self.shuulrich.mdot_infall
            return self.shuulrich.tdisk
        else:
            self.shuulrich.compute_sigdot_and_mdotcap(time,usecumul=True)
            self.infall_mdotcap   = self.shuulrich.mdot_cap
            self.infall_mdot      = self.shuulrich.mdot
            self.infall_sigdot    = self.shuulrich.sigdot
            self.infall_rcentr    = self.shuulrich.rcentr

    def gravinstab_flattening(self,qcrit=2.0,flushgridedges=False):
        """
        In regions where Q_Toomre<2, redistribute the gas surface density 
        such that mass and angular momentum both remain conserved. This
        is done instantly. The idea behind this is that when Q_Toomre<2
        the disk quickly becomes gravitationally unstable and will redistribute
        the gas in the disk to restore Q_Toomre>=2. 

        WARNING: It can happen that the flattening moves stuff all the way
        to the inner edge of the grid (or, for that matter, to the outer
        edge of the grid). This then leads to a pile-up of mass in that cell.
        The cell will then become unphysically massive (i.e. gravitationally 
        unstable). This could lead to a dangerous reversal of mass flow in
        the inner edge. You can either solve this issue by hand, or you
        can set flushgridedges=True, which will put excess mass from the
        inner edge onto the star (and recompute self.omk etc), and remove
        mass from the outer edge. It will then also set a variable 
        self.gravinst_flushfactor_in and self.gravinst_flushfactor_out
        for use with the dust. 
        """
        qcrite   = qcrit * ( 1.0 + 1e-10 )
        sigcrite = self.omk * self.cs / (np.pi * nc.GG * qcrite )
        dsurf    = self.get_dsurf()
        m_tot    = (self.sigma*dsurf).sum()
        l_tot    = (self.sigma*self.omk*self.r**2*dsurf).sum()
        self.compute_qtoomre()
        mask     = np.where(self.qtoomre<qcrit)[0]
        if len(mask)>0:
            if mask[0]==0:
                mask = mask[1:]
        if len(mask)>0:
            if mask[-1]==len(self.r)-1:
                mask = mask[:-1]
        itr      = 0
        sigmaorig= self.sigma.copy()
        self.gravinst_ifrom = []
        self.gravinst_ito   = []
        self.gravinst_frac  = []
        while len(mask)>0 and itr<4:
            itr += 1
            ii = np.where((mask[1:]-mask[:-1])>1)[0]
            if len(ii)>0:
                mask = mask[:ii[0]+1]
            # Compute how much mass and angular momentum is "too much" in this region
            m_ex   = ((self.sigma[mask]-sigcrite[mask])*dsurf[mask]).sum()
            l_ex   = ((self.sigma[mask]-sigcrite[mask])*self.omk[mask]*self.r[mask]**2*dsurf[mask]).sum()
            ex_ratio = l_ex/m_ex
            mleft  = m_ex
            # Now redistribute such that mass and angular momentum are conserved,
            # and Q does not <qcrit
            irmin  = mask.min()-1
            irmax  = mask.max()+1
            while mleft>0.:
                l_in   = self.omk[irmin]*self.r[irmin]**2
                l_out  = self.omk[irmax]*self.r[irmax]**2
                assert ex_ratio>l_in, "Error in angular momentum"
                assert ex_ratio<l_out, "Error in angular momentum"
                x      = (ex_ratio-l_in)/(l_out-ex_ratio)
                assert x>0, "X out of range"
                goahead = True  # This part with the goahead is important if there are multiple zones
                if irmin>0:
                    dmin  = (sigcrite[irmin]-self.sigma[irmin])*dsurf[irmin]
                    if dmin<=0:
                        goahead = False
                        irmin -= 1
                else:
                    dmin  = 1e50
                if irmax<len(self.r)-1:
                    dmout = (sigcrite[irmax]-self.sigma[irmax])*dsurf[irmax]
                    if dmout<=0:
                        goahead = False
                        irmax += 1
                else:
                    dmout = 1e50
                if goahead:
                    if (dmout/dmin)>x:
                        # Fill inner free annulus, if still enough mass
                        if dmin*(1+x)<mleft:
                            # Fill
                            self.sigma[irmin] += dmin/dsurf[irmin]
                            self.sigma[irmax] += dmin*x/dsurf[irmax]
                            mleft          -= dmin*(1+x)
                        else:
                            # Only fill what is left
                            self.sigma[irmin] += mleft/(1+x)/dsurf[irmin]
                            self.sigma[irmax] += mleft*x/(1+x)/dsurf[irmax]
                            mleft           = 0.
                        if irmin>0:
                            irmin -= 1
                    else:
                        # Fill outer free annulus, if still enough mass
                        if dmout*(1+1/x)<mleft:
                            # Fill
                            self.sigma[irmin] += dmout/x/dsurf[irmin]
                            self.sigma[irmax] += dmout/dsurf[irmax]
                            mleft          -= dmout*(1+1/x)
                        else:
                            # Only fill what is left
                            self.sigma[irmin] += mleft/x/(1+1/x)/dsurf[irmin]
                            self.sigma[irmax] += mleft/(1+1/x)/dsurf[irmax]
                            mleft           = 0.
                        if irmax<len(self.r)-1:
                            irmax += 1
            self.sigma[mask] = sigcrite[mask]
            self.compute_qtoomre()
            mask = np.where(self.qtoomre<qcrit)[0]
            if len(mask)>0:
                if mask[0]==0:
                    mask = mask[1:]
            if len(mask)>0:
                if mask[-1]==len(self.r)-1:
                    mask = mask[:-1]
        assert itr<4, "GI Flattening not converged"
        masserr = np.abs((self.sigma*dsurf).sum()/(sigmaorig*dsurf).sum()-1.0)
        assert masserr<1e-10, "GI Flattening: mass conservation problem: Err = {}".format(masserr)
        if itr>0:
            # Make a list of mass movements, to be used for the dust 'flattening'
            # Could also be used for chemistry.
            mask_gain = np.where(self.sigma>sigmaorig)[0]
            mask_loss = np.where(self.sigma<sigmaorig)[0]
            k = 0
            dm_gain  = (self.sigma[mask_gain[k]]-sigmaorig[mask_gain[k]])*dsurf[mask_gain[k]]
            for i in range(len(mask_loss)):
                m_orig   = sigmaorig[mask_loss[i]]*dsurf[mask_loss[i]]
                dm_loss  = (sigmaorig[mask_loss[i]]-self.sigma[mask_loss[i]])*dsurf[mask_loss[i]]
                while dm_gain<dm_loss:
                    self.gravinst_ifrom.append(mask_loss[i])
                    self.gravinst_ito.append(mask_gain[k])
                    self.gravinst_frac.append(dm_gain/m_orig)
                    dm_loss -= dm_gain
                    k       += 1
                    if k<len(mask_gain):
                        dm_gain  = (self.sigma[mask_gain[k]]-sigmaorig[mask_gain[k]])*dsurf[mask_gain[k]]
                    else:
                        dm_gain  = dm_loss
                        k        = len(mask_gain)-1
                self.gravinst_ifrom.append(mask_loss[i])
                self.gravinst_ito.append(mask_gain[k])
                self.gravinst_frac.append(dm_loss/m_orig)
                dm_gain -= dm_loss
        if(hasattr(self,'gravinst_flushfactor_in')): delattr(self,'gravinst_flushfactor_in')
        if(hasattr(self,'gravinst_flushfactor_out')): delattr(self,'gravinst_flushfactor_out')
        if flushgridedges:
            if self.sigma[0]>sigcrite[0]:
                dmstar = (self.sigma[0]-sigcrite[0])*dsurf[0]
                self.gravinst_flushfactor_in = dmstar/(self.sigma[0]*dsurf[0])  # Necessary for dust
                self.sigma[0] = sigcrite[0]
                mstar = self.mstar + dmstar
                self.update_stellar_mass(mstar)
            if self.sigma[-1]>sigcrite[-1]:
                dmout = (self.sigma[-1]-sigcrite[-1])*dsurf[-1]
                self.gravinst_flushfactor_out = dmout/(self.sigma[-1]*dsurf[-1])  # Necessary for dust
                self.sigma[-1] = sigcrite[-1]
        self.compute_qtoomre()
                
                    
    def plot(
            self, quantity, ylabel=None, oplot=False, xlog=True, ylog=True,
            xmin=None, xmax=None, ymin=None, ymax=None, **kwargs):
        """
        A special-for-disk plotting routine. The x-coordinate is always disk.r in au.
        Plotted is the array 'quantity'.

        ARGUMENTS:
          quantity           Array of quantity(r) to be plotted

        OPTIONAL ARGUMENTS/KEYWORDS:
          ylabel             The string to put on the y-axis
          oplot              If False, then this routine will plt.clf(), otherwise not.
          xlog, ylog         Make the x, y axis logarithmic
          xmin, xmax         Limits on x
          ymin, ymax         Limits on y

        EXAMPLE:

        from disklab.diskradial import *
        from disklab.natconst import *
        d=DiskRadialModel()
        d.make_disk_from_simplified_lbp(1e1,10*au,1.0)
        d.plot(d.sigma,ylabel=r'$\Sigma_{\mathrm{gas}}$')

        """
        import matplotlib.pyplot as plt
        if not oplot:
            plt.clf()
        if len(quantity) == len(self.r):
            x = self.r / nc.au
        elif len(quantity) == len(self.r) - 1:
            x = self.get_ri() / nc.au
        else:
            raise ValueError('Array length does not match radial grid')
        if ylog:
            plt.plot(x, np.abs(quantity), **kwargs)
        else:
            plt.plot(x, quantity, **kwargs)
        plt.xlabel('r [au]')
        if ylabel is not None:
            plt.ylabel(ylabel)
        if not oplot:
            if xlog:
                plt.xscale('log')
            else:
                plt.xscale('linear')
            if ylog:
                plt.yscale('log')
            else:
                plt.yscale('linear')
        if xmin is not None:
            plt.xlim(left=xmin)
        if xmax is not None:
            plt.xlim(right=xmax)
        if ymin is not None:
            plt.ylim(bottom=ymin)
        if ymax is not None:
            plt.ylim(top=ymax)

    def anim(
            self, time, quantity, ifig=None, ylabel=None, xlog=True, ylog=True,
            xmin=None, xmax=None, ymin=None, ymax=None, interval=100, pause=10, **kwarg):
        """
        A special-for-disk animation routine for plotting the evolution of
        an r-dependent quantity with time. See the plot() routine above.
        The only difference is that 'quantity' is now a 2D array (time and
        space), and an array 'time' has to be given.

        ARGUMENTS:
          time[0:nt]            Array of times of the snapshots in seconds
          quantity[0:nt,0:nr]   Array of the quantity to be plotted, first
                                index is time, second is radius.

        OPTIONAL ARGUMENTS/KEYWORDS:
          interval           The time lag between frames
          pause              The pause at the end, before the animation restarts (in units of frames)
          ylabel             The string to put on the y-axis
          oplot              If False, then this routine will plt.clf(), otherwise not.
          xlog, ylog         Make the x, y axis logarithmic
          xmin, xmax         Limits on x
          ymin, ymax         Limits on y

        EXAMPLE:

        from disklab.diskradial import *
        from disklab.natconst import *
        d=DiskRadialModel(rout=1000*nc.au)
        d.make_disk_from_simplified_lbp(1e2,1*au,1.0)
        tmax = 1e6*year
        nt   = 100
        time = np.linspace(0,tmax,nt)
        sigma_array = np.zeros((nt,len(d.r)))
        sigma_array[0,:] = d.sigma.copy()
        for itime in range(1,nt):
            dt = time[itime]-time[itime-1]
            d.compute_viscous_evolution_next_timestep(dt)
            sigma_array[itime,:] = d.sigma.copy()
        d.anim(time,sigma_array,ymin=1e-5,pause=30)
        plt.show()

        """
        import matplotlib.animation as animation
        import matplotlib.pyplot as plt
        self.itime = 0
        self.time  = time.copy()
        self.quantity = quantity
        self.pause = pause

        def animupdate(frameNum, disk, a0, txt0):
            iitime = disk.itime + 0
            ntime = len(disk.time)
            if iitime > ntime - 1:
                iitime = ntime - 1
            y = disk.quantity[iitime, :]
            a0.set_data(disk.r / nc.au, y)
            txt0.set_text('t = {0:9.2e} Myr'.format(disk.time[iitime] / (1e6 * nc.year)))
            disk.itime = (disk.itime + 1) % (ntime + disk.pause)
        if ifig is None:
            fig = plt.figure()
        else:
            fig = plt.figure(ifig)
            plt.clf()
        if xmin is None:
            xmin = self.r.min() / nc.au
        if xmax is None:
            xmax = self.r.max() / nc.au
        if ymin is None:
            ymin = quantity.min()
        if ymax is None:
            ymax = quantity.max()
        ax  = plt.axes(xlim=(xmin, xmax), ylim=(ymin, ymax))
        plt.xlabel('r [au]')
        if ylabel is not None:
            plt.ylabel(ylabel)
        if hasattr(self, 'myanimation'):
            self.myanimation._stop()
            del self.myanimation
        if xlog:
            plt.xscale('log')
            xtxt = xmin * np.exp((np.log(xmax) - np.log(xmin)) * 0.7)
        else:
            plt.xscale('linear')
            xtxt = xmin + (xmax - xmin) * 0.7
        if ylog:
            plt.yscale('log')
            ytxt = ymin * np.exp((np.log(ymax) - np.log(ymin)) * 0.9)
        else:
            plt.yscale('linear')
            ytxt = ymin + (ymax - ymin) * 0.9
        a0, = ax.plot([], [])
        txt0 = ax.text(xtxt, ytxt, 't = {0:9.2e} Myr'.format(time[0] / (1e6 * nc.year)))
        self.myanimation = animation.FuncAnimation(
            fig, animupdate, fargs=(self, a0, txt0), interval=interval)
    
    def compute_torque_paarde(self, mpl=nc.Mea, adi_gamma=7/5, bhpr=0.4, normed=True, components=False):
        """
        Computes the torques acting on a planet embedded in a disk according to Paardekooper et al. (2011).
        They assume that the torque is given by the radial gradients in the disk(temperature, surface density (vortensity) 
        and entropy). The formula was tested and confirmed with 2D simulations (r, z). 

        A linear Lindblad torque and (possibly) non-linear corotation torques are assumed and computed.
        The corotation torque is divided into two parts: 
        -a barotropic part which is produced by the conservation of vortensity by the material.
        -an entropy related part which "is exerted by density structures produced by material conserving
         its entropy, plus an additional component component linked to the production of specific vorticity
         at the outgoing separatrices"
        
        To compute the torque, the disk needs surface density and a mean Rosseland opacity. If 
        rhomid or the viscosity nu is not given, they are computed using self.compute_rhomid_from_sigma()
        and self.compute_nu() respectively. 

        The power law coefficients of tmid and sigma are computed and stored as self.plsig and self.pltmid. 
        The computation is done by taking the logarithmic differential quotient. The boundary terms are taken as
        the boundary terms of the differential quotient, the inner terms are taken as the mean of the differential quotien, 
        of the cells left and right of it. 

        Additionally, omega is computed using self.compute_omega(). 
        The thermal diffusion coefficient chi is saved as self.chi. The saturation parameters p_chi and p_nu are also
        stored. 

        Stored torques:
        At last, the Lindblad torque is saved as a 1D array as self.gamma_L 
        
        Other torques are saved as 2D arrays, where the rows represent the radial grid and the columns the 
        planet masses.

        self.gamma_0
        self.gamma_c_baro 
        self.gamma_c_ent  
        self.gamma_tot    

        Paramerters:
        mpl         = Mass(es) of the planet(s). Can be given as a 1D array.
        adi_gamma   = adiabatic coefficient
        bhpr        = fraction between the scale-free smoothing length b of the gravitation potential and hpr (H/r).
        
        If normed = False, the torques will be given in cgs units. Otherwise they are normed by Gamma_0, 
        where Gamma_0 = (q/h)**2*Sigma_p*r_p**4*Omega_p**2.
        
        If components = True, the individual components of the entropy related and barotropic torque are also stored
        as
        self.gamma_c_lin_baro 
        self.gamma_hs_baro 
    
        self.gamma_hs_ent 
        self.gamma_c_lin_ent     
        where c_lin represents the linear approximation of the corotation torque and hs the non-linear horseshoe drag.

        (Written and implemented by Andrej Hermann, July 2019)
        """
        #-----------------------requirements---------------------#
        assert hasattr(self, 'mean_opacity_rosseland'), 'The mean Rosseland opacity is needed for the computation of the torque.'
        assert hasattr(self, 'sigma'), 'Surface density sigma is needed for the computation of the torque'

        if not hasattr(mpl, '__len__'):
            mpl = np.array([mpl])
        
        if not hasattr(self, 'rhomid'):
            self.compute_rhomid_from_sigma()
        
        if not hasattr(self, 'nu'):
            self.compute_nu()
        
        dlntdlnr = (np.log(self.tmid[1:])-np.log(self.tmid[:-1]))/(np.log(self.r[1:])-np.log(self.r[:-1]))
        self.pltmid = np.hstack((dlntdlnr[0], (dlntdlnr[1:]+dlntdlnr[:-1])/2, dlntdlnr[-1]))                        #power law coefficient of tmid

        dlnsigmadlnr = (np.log(self.sigma[1:])-np.log(self.sigma[:-1]))/(np.log(self.r[1:])-np.log(self.r[:-1]))
        self.plsig = np.hstack((dlnsigmadlnr[0], (dlnsigmadlnr[1:]+dlnsigmadlnr[:-1])/2, dlnsigmadlnr[-1]))         #power law coefficient of sigma

        self.compute_omega()

        if self.omega[-1]<0:
            self.rhomid[-1] = self.rhomid[-2]       #sometimes self.omega[-1] becomes negative and causes a cascade of errors. 
            self.omega[-1] = self.omega[-2]         #To avoid this, we estimate rhomid[-1] and omega[-1] to be equal to their values at [-2]

        q       = mpl/self.mstar                    #mass ratio of the planet to the star

        hpr = self.hp/self.r                        #scale free aspect ratio of the disk 
        
        #-----------------------saturation parameters---------------------#
        self.chi   =  (16* adi_gamma * (adi_gamma-1) * nc.ss * self.tmid**4 / 
                    (3 * self.mean_opacity_rosseland * self.rhomid**2 * self.hp**2 * self.omega**2))    #corrected thermal diffusion coefficient (Bitsch & Kley 2011)

        Q       = np.where(self.chi > 1e13, 2/3 * self.chi * self.omega * hpr**-1 * self.cs**-2, 1e-3)  #to account for the difference on the Lindblad torque
        g_eff   = np.where(self.chi > 1e13,                                                             #between high and low thermal diffusivity (isothermal and 
                2*Q*adi_gamma / (adi_gamma*Q + 0.5*np.sqrt(2*                                           #adiabatic disks), an effective gamma (g_eff) is computed
                np.sqrt((adi_gamma**2*Q**2+1)**2-16*Q**2*(adi_gamma-1)) +2*adi_gamma**2*Q**2-2) ),      
                adi_gamma)

        xi      = -self.pltmid+(adi_gamma-1)*self.plsig                                                 #power law coefficient of the entropy 
        x_s     = np.tensordot(np.sqrt(q), 1.1/g_eff**0.25 * (0.4/bhpr)**(-0.25)*np.sqrt(1/hpr), axes=0)    #half-width of the horseshoe region in units of r 

        self.p_nu   =   2/3*self.r*np.sqrt(self.omega*0.5*x_s**3/nc.pi/self.nu)     #saturation parameter associated with viscosity nu PB (19)
        self.p_chi  =   self.r*np.sqrt(self.omega*0.5*x_s**3/nc.pi/self.chi)        #saturation parameter associated with thermal diffusivity chi PB (40)

        #-----------------------saturation and transition functions---------------------#
        def F(p):
            return 1 / ( 1 + (p/1.3)**2 )       #saturation function. ~1 for low p (high diffusion) and goes to 0 for high p (low diffusion)

        def G(p):
            frac = 8/45 / np.pi
            condition = p < np.sqrt( frac )
            return np.where(condition, 16/25 * frac**(-0.75) * p**1.5,
                    1 - 9/25 * frac**(4/3) * p**(-8/3))
                                                            #transition functions for the transition from linear torque to non-linear horseshoe drag 
                                                            #according to Gamma_hs * G + (1-K) * Gamma_lin
        def K(p):
            frac = 28/45 / np.pi
            condition = p < np.sqrt( frac )
            return np.where(condition, 16/25 * frac**(-0.75) * p**1.5,
                        1 - 9/25 * frac**(4/3) * p**(-8/3))
        #-----------------------torque computation---------------------#

        self.gamma_0        =   np.tensordot(q**2, (1/hpr)**2*self.sigma*self.r**4*self.omega**2, axes=0)                  

        self.gamma_L        =   (-2.5+1.7*self.pltmid-0.1*self.plsig)/g_eff         #linear Lindsblad-Torque (normalized) PB (3)
        
        gamma_hs_ent        =   7.9 * xi / g_eff**2                                 #entropy-related part of the horseshoe drag (normalized). PB (5)
        gamma_c_lin_ent     =   (2.2 - 1.4/g_eff) * xi/g_eff                        #entropy-related part of the linear corotation torque (normalized). PB (7)
        self.gamma_c_ent    =   (
                                gamma_hs_ent * F(self.p_nu)*F(self.p_chi) * np.sqrt( G(self.p_nu)*G(self.p_chi) ) +
                                np.sqrt( (1 - K(self.p_nu))*(1 - K(self.p_chi)) ) * gamma_c_lin_ent
                                )                                                   #total non-barotropic, entropy-related corotation torque (normalized) PB (53)

        gamma_hs_baro       =   1.1*(1.5+self.plsig)/g_eff                          #barometric part of the horseshoe drag (normalized). PB (4)
        gamma_c_lin_baro    =   0.7*(1.5+self.plsig)/g_eff                          #barotropic part of the linear corotation torque (normalized). PB (6)
        self.gamma_c_baro   =   (
                                gamma_hs_baro * F(self.p_nu)*G(self.p_nu) + 
                                (1-K(self.p_nu)) * gamma_c_lin_baro
                                )                                                   #total barotropic part of the horseshoe drag (normalized) PB(52)

        self.gamma_tot      = self.gamma_L + self.gamma_c_baro + self.gamma_c_ent   #total torque (normalized)

        if not normed:
            self.gamma_L            *=  self.gamma_0
            self.gamma_c_baro       *=  self.gamma_0
            self.gamma_c_ent        *=  self.gamma_0
            self.gamma_tot          *=  self.gamma_0

        if components:
            self.gamma_hs_baro      =  gamma_hs_baro    *(self.gamma_0)**(1-normed)
            self.gamma_hs_ent       =  gamma_hs_ent     *(self.gamma_0)**(1-normed)
            self.gamma_c_lin_baro   =  gamma_c_lin_baro *(self.gamma_0)**(1-normed)
            self.gamma_c_lin_ent    =  gamma_c_lin_ent  *(self.gamma_0)**(1-normed)
    
    def compute_migration_timescale(self, mpl, normed=True):
        """
        Compute the migration timescale for circular orbits according to Kley & Nelson (2012) Eq. (14).
        If the torque is not normed to Gamma_0, assign normed=False.
        """
        assert hasattr(self, 'gamma_tot'), 'Total torque has to be computed.'

        J_p = np.tensordot(mpl, np.sqrt(nc.GG*self.mstar*self.r), axes=0)
        self.tau_mig = 0.5*J_p/self.gamma_tot/self.gamma_0**normed


# ---------------------------------------------------------------------------------------------

class ShuUlrichModel(object):
    """
    Shu-Ulrich cloud collapse model. We follow the papers by Shu 1977 ApJ 214,
    488, and Ulrich 1976 ApJ 210, 377 and combine them into one. This will allow
    us to accrete matter onto the disk in the way that is described in Hueso &
    Guillot 2005 A&A 442, 703, and later by Dullemond, Natta & Testi 2006 A&A
    645, 69 and Dullemond, Apai & Walch 2006 A&A 640, 67.

    Note that what is usually called _the_ Shu model is the "expansion wave
    collapse solution" for A=2 and m0=0.975, which is a limiting case of a family
    of Shu model solutions. The cases for A>2 and m0>0.975 assume that the 
    gas velocity at r=\infty the gas velocity is not zero, but inward pointing.
    This model collapses faster: higher accretion rate, shorter time. A full
    solution for A>2 and m0>0.975 would, however, require numerical integration. 
    Here we allow, instead, to set m0 to a value >0.975, to mimick the faster
    accretion (though this solution is then approximate). Setting m0 to, e.g.,
    2.0 would make Mdot roughly twice as high, but the time of collapse twice
    as short. The cloud initial radius, however, stays the same. Also the 
    angular momentum stuff stays the same.

    ARGUMENTS:
      mcloud        = Initial cloud mass [g]
      tcloud        = Cloud temperature [K]
      omcloud       = Cloud rotation rate [1/s]
      idistr        = (optional): ==1 Use modified infall (Hueso & Guillot 2005)
      rgrid         = (optional): If set, then this will be the radial grid 
                      on which the infall rate sigmadot will be computed.
                      Also the self.tdisk is computed, which is the first
                      time the Ulrich infall has sufficient angular momentum
                      to reach the inner edge of the rgrid.
    
    """
    def __init__(self,xmn=1e-6,nx=1000,mcloud=None,tcloud=None,omcloud=None,
                 idistr=1,rgrid=None,m0=0.975,mu=2.3):
        assert mcloud is not None, "Must specify mcloud"
        assert tcloud is not None, "Must specify tcloud"
        assert omcloud is not None, "Must specify omcloud"
        self.mcloud         = mcloud
        self.tcloud         = tcloud
        self.omcloud        = omcloud
        self.mu             = mu
        nx2                 = nx//2
        x                   = xmn * (1./xmn)**np.linspace(0,1,nx2)
        x                   = np.hstack((0.5*x[:-1],1.-0.5*x[::-1]))
        xi                  = 0.5 * (x[1:] + x[:-1])
        xi                  = np.hstack((0,xi,1))
        self.r_dimless      = x
        self.ri_dimless     = xi
        self.dsurf_dimless  = np.pi*(xi[1:]**2-xi[:-1]**2)
        if(idistr==0):
            #
            # Distribute the matter over the disk in the Ulrich way
            #
            self.sigdot_dimless = 1/(x*np.sqrt(1.0-x))
        elif(idistr == 1):
            #
            # Distribute the matter over the disk in such a way that
            # there is no friction of the infalling matter with the disk
            # (see Hueso & Guillot 2005)
            #
            self.sigdot_dimless = 1/(x**1.5*np.sqrt(1-np.sqrt(x)))
        else:
            raise ValueError("Do not know idistr value")
        self.mdot_dimless_cum  = (self.sigdot_dimless*self.dsurf_dimless).cumsum()
        self.mdot_dimless_cum  = np.hstack((0,self.mdot_dimless_cum))
        self.sigdot_dimless   /= self.mdot_dimless_cum[-1]
        self.mdot_dimless_cum /= self.mdot_dimless_cum[-1]
        cs                  = np.sqrt(nc.kk*tcloud/(self.mu*nc.mp))
        mdot_infall         = m0 * cs**3 / nc.GG
        rcloud              = nc.GG * mcloud / (2 * cs**2)
        twave               = rcloud / cs
        twavem              = twave * (2. / m0)
        self.cs             = cs
        self.mdot_infall    = mdot_infall
        self.rcloud         = rcloud
        self.twave          = twave
        self.twavem         = twavem
        if rgrid is not None:
            self.r = rgrid
            ri     = 0.5 * (self.r[1:] + self.r[:-1])
            ri     = np.hstack((self.r[0],ri,self.r[-1]))
            self.ri = ri
            self.dsurf = np.pi*(ri[1:]**2-ri[:-1]**2)
            self.sigdot = np.zeros_like(rgrid)
            self.tdisk = ( 16 * self.r[0] / ( omcloud**2 * cs * m0**3 ) )**0.33333

    def compute_rcentr_and_mdot(self,time):
        m0 = 0.975
        self.time = time
        if(time < self.twavem):
            self.mcentr = (m0/2.) * self.mcloud * (time / self.twave)
            self.mdot   = self.mdot_infall
        else:
            self.mcentr = self.mcloud
            self.mdot   = 0
        rcoll  = self.cs * time
        rcoll0 = rcoll * (m0/2.0)
        if(time > self.twave):
            rcoll = self.rcloud
        if(time > self.twavem):
            rcoll0 = self.rcloud
        self.rcentr = self.omcloud**2 * rcoll0**4 / (nc.GG * (self.mcentr + 1.0)) + 1e-99

    def compute_sigdot_and_mdotcap(self,time,usecumul=True):
        from scipy.interpolate import interp1d
        r  = self.r
        ri = self.ri
        self.compute_rcentr_and_mdot(time)
        assert self.rcentr<self.r[-1], "Error in Ulrich: Centrifugal radius larger than max of radial grid."
        self.mdot_cap  = 0.
        self.mdot_disk = 0.
        self.sigdot[:] = 0.
        if self.mdot>0:
            x0 = r[0]/self.rcentr
            if x0<1:
                dumf = interp1d(self.ri_dimless,self.mdot_dimless_cum,kind='cubic')
                self.mdot_cap  = self.mdot * dumf(x0)
                self.mdot_disk = self.mdot - self.mdot_cap
                iimlr = np.where(ri < self.rcentr)[0]
                imlr  = iimlr[:-1]
                xi    = ri[iimlr]/self.rcentr
                if usecumul:
                    mdc = dumf(xi)
                    mdc = np.hstack((mdc,1.))
                    xi  = np.hstack((xi,1.))
                    md  = mdc[1:]-mdc[:-1]
                    self.sigdot[iimlr] = self.mdot * md / self.dsurf[iimlr]
                else:
                    self.sigdot[imlr] = ( self.mdot_disk / self.rcentr**2 ) * np.interp(x,self.r_dimless,self.sigdot_dimless)
                factor = self.mdot_disk / (self.sigdot*self.dsurf).sum()
                assert np.abs(factor-1.)<1e-2, "Mass conservation in Ulrich violated"
            else:
                self.mdot_cap  = self.mdot
                self.mdot_disk = 0

# ---------------------------------------------------------------------------------------------

class DiskRadialComponent(object):
    """
    To allow the disk to contain one or more dust or chemical components, this class contains
    all the methods needed for moving such a component around (e.g. dust drift, mixing etc).
    """
    def __init__(self, diskradialmodel, sigma=None, agrain=None, xigrain=None, St=None):
        assert hasattr(diskradialmodel, 'sigma'), "Cannot create a dust component if the disk model does not yet have sigma."
        self.diskradialmodel = diskradialmodel     # Link to the parent disk model
        self.r         = diskradialmodel.r    # Link radial grid to disk model for convenience
        self.omk       = diskradialmodel.omk  # Link Omega_K to disk model for convenience
        nr = len(self.r)
        if sigma is None:
            self.sigma = np.zeros(len(self.r))
        else:
            self.sigma   = copy.copy(sigma)    # Not a link, but an independent copy
        if agrain is not None:
            self.agrain  = np.zeros(nr) + copy.copy(agrain)   # Not a link, but an independent copy
        if xigrain is not None:
            self.xigrain = np.zeros(nr) + copy.copy(xigrain)  # Not a link, but an independent copy
        else:
            self.xigrain = np.zeros(nr) + 3.0  # Default dust material density
        if St is not None:
            self.St      = np.zeros(nr) + copy.copy(St)       # Not a link, but an independent copy
        if St is not None:
            self.compute_agrain_from_stokes()
            self.compute_hdust_and_rhodust()
        if agrain is not None:
            self.compute_stokes_from_agrain()
            self.compute_hdust_and_rhodust()
        if agrain is not None:
            self.grain   = GrainModel(agrain=agrain, xigrain=xigrain)
            self.grain.diskradialcomp = self

    def compute_mass(self):
        """
        (Dust version)
        Compute the dust total mass by numeric integration. Will be put in self.mass.
        """
        ds = nc.pi * (self.r[1:]**2 - self.r[:-1]**2)
        sg = 0.5 * (self.sigma[1:] + self.sigma[:-1])
        dm = sg * ds
        self.mass = dm.sum()

    def compute_qtoomre(self):
        """
        (Dust version)
        Compute the Toomre Q value for this dust component.
        """
        assert hasattr(self, 'sigma'), "Error: sigma not set"
        assert hasattr(self, 'hp'), "Error: dust vertical height hp not set"
        hp     = self.hp
        omk    = self.omk
        sigma  = self.sigma
        self.qtoomre = hp * omk**2 / (nc.pi * nc.GG * sigma)

    def compute_mgrain_from_agrain(self):
        """
        Compute the grain mass everywhere.
        """
        assert hasattr(self, 'agrain'), "Error: no agrain present; cannot compute grain mass"
        assert hasattr(self, 'xigrain'), "Error: no xigrain present; cannot compute grain mass"
        nr = len(self.r)
        if not hasattr(self, 'mgrain'):
            self.mgrain  = np.zeros(nr)
        agrain  = np.zeros(nr) + self.agrain
        xigrain = np.zeros(nr) + self.xigrain
        self.mgrain[:] = (4 * nc.pi / 3.) * xigrain * agrain**3

    def compute_dustmass(self):
        """
        Compute the total mass in dust in the disk, from self.sigdust
        """
        ds = nc.pi * (self.r[1:]**2 - self.r[:-1]**2)
        sg = 0.5 * (self.sigma[1:] + self.sigma[:-1])
        dm = sg * ds
        self.mass = dm.sum()

    def compute_stokes_from_agrain(self, keepmass=False, dv=1e3):
        """
        This routines uses the GrainModel() class to compute the stopping time
        and the Stokes number from the agrain array. If keepmass==True, then
        the grain mass is not recomputed from the agrain. If dv is set to
        either a value or an array (in cm/s) then this value is used as
        the relative velocity between the dust and the gas. This only matters
        for grains that are in the Stokes regime (i.e.\ grains larger than
        the gas mean free path).
        """
        if hasattr(self, 'grain'):
            assert np.abs(self.agrain / self.grain.agrain - 1.0).max() < 1e-6, 'Inconsistency: dust component has different agrain than its grain model.'
            assert np.abs(self.xigrain / self.grain.xigrain - 1.0).max() < 1e-6, 'Inconsistency: dust component has different xigrain than its grain model.'
        nr          = len(self.r)
        agrain      = np.zeros(nr) + self.agrain
        dv          = np.zeros(nr) + dv
        xigrain     = np.zeros(nr) + self.xigrain
        if not hasattr(self, 'tstop'):
            self.tstop  = np.zeros(nr)
        if not hasattr(self, 'St'):
            self.St     = np.zeros(nr)
        if not hasattr(self, 'mgrain'):
            self.mgrain = np.zeros(nr)
        if np.isscalar(self.tstop):
            self.tstop  = np.zeros(nr)
        if np.isscalar(self.St):
            self.St     = np.zeros(nr)
        if np.isscalar(self.mgrain):
            self.mgrain = np.zeros(nr)
        if not keepmass:
            self.compute_mgrain_from_agrain()
        mgrain      = np.zeros(nr) + self.mgrain
        for ir in range(nr):
            g        = GrainModel(agrain=agrain[ir], xigrain=xigrain[ir])
            g.mgrain = mgrain[ir]
            # r        = self.r[ir]
            rhomid   = self.diskradialmodel.rhomid[ir]
            tgas     = self.diskradialmodel.tmid[ir]
            g.compute_tstop(rhomid, tgas, dv[ir])
            self.St[ir]    = self.omk[ir] * g.tstop
            self.tstop[ir] = g.tstop

    def compute_agrain_from_stokes(self, dv=1e3):
        """
        This routines uses the GrainModel() class to compute the grain size
        and stopping time from the St array. If dv is set to
        either a value or an array (in cm/s) then this value is used as
        the relative velocity between the dust and the gas. This only matters
        for grains that are in the Stokes regime (i.e.\ grains larger than
        the gas mean free path).
        """
        if hasattr(self, 'grain'):
            assert np.abs(self.agrain / self.grain.agrain - 1.0).max() < 1e-6, 'Inconsistency: dust component has different agrain than its grain model.'
            assert np.abs(self.xigrain / self.grain.xigrain - 1.0).max() < 1e-6, 'Inconsistency: dust component has different xigrain than its grain model.'
        nr          = len(self.r)
        St          = self.St
        dv          = np.zeros(nr) + dv
        xigrain     = np.zeros(nr) + self.xigrain
        if not hasattr(self, 'tstop'):
            self.tstop  = np.zeros(nr)
        if not hasattr(self, 'agrain'):
            self.agrain = np.zeros(nr)
        if not hasattr(self, 'mgrain'):
            self.mgrain = np.zeros(nr)
        if np.isscalar(self.tstop):
            self.tstop  = np.zeros(nr)
        if np.isscalar(self.agrain):
            self.agrain = np.zeros(nr)
        if np.isscalar(self.mgrain):
            self.mgrain = np.zeros(nr)
        self.tstop[:]  = St / self.omk
        for ir in range(nr):
            # r       = self.r[ir]
            rhomid  = self.diskradialmodel.rhomid[ir]
            tgas    = self.diskradialmodel.tmid[ir]
            d       = GrainModel(tstop=self.tstop[ir], xigrain=xigrain[ir])
            d.solve_agrain(rhomid, tgas, dv[ir])
            self.agrain[ir] = d.agrain
            d.compute_mgrain()
            self.mgrain[ir] = d.mgrain

    def compute_hdust_and_rhodust(self):
        """
        Given the Stokes number of the dust, the turbulence of the
        gas and the surface density and temperature of the gas, the vertical
        height of the dust layer can be computed with this method.
        """
        nr = len(self.r)
        assert hasattr(self, 'St'), "Error: no Stokes number of the dust computed"
        assert hasattr(self.diskradialmodel, 'hp'), "Error: no gas vertical scale height hp present"
        assert hasattr(self.diskradialmodel, 'alpha'), "Error: no alpha viscosity/diffusion parameter given"
        if not hasattr(self, 'hp'):
            self.hp      = np.zeros(nr)
        if not hasattr(self, 'rhomid'):
            self.rhomid  = np.zeros(nr)
        # Compute dust scale height: Eq. 51 of Birnstiel et al. 2010.
        St              = self.St
        self.hp[:]      = self.diskradialmodel.hp * np.minimum((self.diskradialmodel.alpha / (np.minimum(St, 0.5) * (1 + St**2)))**0.5, 1.)
        self.rhomid[:]  = self.sigma / (self.hp * (2 * nc.pi)**0.5)

    def compute_dustvr_at_interfaces(self, alphamodel=True, fixgas=False):
        """
        Compute the radial drift velocity of the dust at the interfaces. We follow
        Birnstiel, Dullemond & Brauer (2010) A&A 513, 79.

        ARGUMENTS:
        alphamodel  = If True, then recompute self.nu from alpha-recipe (default)
        fixgas      = If True, then do *not* include the inward gas motion in dust drift.

        ** BEWARE: **
        Always make sure to have updated the midplane density and temperature
        and possibly also the Stokes number (and equiv tstop) before calling
        this subroutine, if you have evolved the gas beforehand.
        """
        assert hasattr(self, 'agrain'), "Cannot compute dust radial drift if not is not yet added."
        #
        # If requested, compute nu from alpha
        #
        if alphamodel:
            self.diskradialmodel.compute_nu()
        #
        # Compute v_K and cs at interfaces
        #
        r     = self.r
        ri    = 0.5 * (r[1:] + r[:-1])
        omki  = (nc.GG * self.diskradialmodel.mstar / ri**3)**0.5
        vki   = omki * ri
        csi   = 0.5 * (self.diskradialmodel.cs[1:] + self.diskradialmodel.cs[:-1])
        #
        # Compute the radial velocity of the gas.
        # No need to recalculate alpha. Use upwind scheme for v_r.
        #
        if fixgas:
            vrgas = 0.0    # Gas radial velocity not included
        else:
            self.diskradialmodel.compute_vr_at_interfaces(alphamodel=False, upwind=True)
            vrgas = self.diskradialmodel.vr
        #
        # Compute the dln(p)/dln(r) at the interfaces by calling the method
        # that computes omega
        #
        self.diskradialmodel.compute_omega(interp=False)
        #
        # Compute the average Stokes number at the interfaces
        #
        Sti    = 0.5 * (self.St[1:] + self.St[:-1])
        #
        # Compute the dust radial drift velocity including the passive advection by the gas
        #
        self.vr = vrgas / (1.0 + Sti**2) + self.diskradialmodel.dlnpdlnr * csi**2 / vki / (Sti + 1.0 / Sti)

    def get_dust_radial_drift_next_timestep(self, dt, alphamodel=True, fixgas=False, extracond=None,
                                            bcin='zerogradient',bcout='zerogradient'):
        """
        Advance the dust component one time step into the future.
        Radial drift and turbulent mixing included, as well as the
        gas drag as the gas is moving inward.

        ARGUMENTS:
        dt          = Time step in seconds
        alphamodel  = If True, then recompute self.nu from alpha-recipe (default)
        fixgas      = If True, then do *not* include the inward gas motion in dust drift.
        extracond   = (for special purposes only) List of extra internal conditions
        bcin,bcout  = String denoting the kind of boundary condition for the dust
                      'zerogradient'   : Default: simply set gradient of Sigma_dust to zero
                      'closed'         : Do not allow dust to cross this border

        Note: If self.diskradialmodel.alphamix is present, then this alpha will be used (instead of the
        usual self.alpha) for the turbulent mixing.

        ** BEWARE: **
        Always make sure to have updated the midplane density and temperature,
        and then call the compute_stokes_from_agrain() before calling this subroutine,
        if you have evolved the gas beforehand.
        """
        #
        # If requested, compute nu and dmix from alpha
        #
        if alphamodel:
            self.diskradialmodel.compute_nu()
            if hasattr(self.diskradialmodel, 'alphamix'):
                self.dmix = self.diskradialmodel.alphamix * self.diskradialmodel.cs * self.diskradialmodel.cs / self.omk / self.diskradialmodel.Sc
            else:
                self.dmix = self.diskradialmodel.nu / self.diskradialmodel.Sc
            self.dmix[:] *= 1.0 / (1.0 + self.St**2)
        #
        # Cast into diffusion equation form
        #
        x    = self.r
        y    = 2 * nc.pi * self.r * self.sigma             # Dust
        g    = 2 * nc.pi * self.r * self.diskradialmodel.sigma   # Gas
        d    = self.dmix
        di   = 0.5 * (d[1:] + d[:-1])
        self.compute_dustvr_at_interfaces(alphamodel=alphamodel, fixgas=fixgas)
        vi   = self.vr
        #
        # Source term only if self.sigdot is given
        #
        if hasattr(self, 'sigdot'):
            s = 2 * nc.pi * self.r * self.sigdot
        else:
            s = np.zeros(len(x))    # For now no source term
        #
        # Set boundary conditions
        #
        if bcin=='zerogradient':
            bcl  = (1, 0, 0, 1)  # Simply set dy/dx=0 at inner edge
        elif bcin=='closed':
            bcl  = (1, 0, 0, -1) # Closed boundary at inner edge
        else:
            raise(ValueError('Inner boundary condition for dust invalid'))
        if bcout=='zerogradient':
            bcr  = (1, 0, 0, 1)  # Simply set dy/dx=0 at outer edge
        elif bcout=='closed':
            bcr  = (1, 0, 0, -1) # Closed boundary at outer edge
        else:
            raise(ValueError('Outer boundary condition for dust invalid'))
        #
        # Get the new value of y after one time step dt
        #
        y    = solvediffonedee(x, y, vi, di, g, s, bcl, bcr, dt=dt,
                               int=True, upwind=True, extracond=extracond)
        #
        # Obtain new sigdust
        #
        sigma = y / (2 * nc.pi * self.r)
        #
        # Return
        #
        return sigma

    def compute_dust_radial_drift_next_timestep(self, dt, alphamodel=True, fixgas=False, extracond=None, updatestokes=True):
        """
        Wrapper routine around dust_radial_drift_next_timestep(), which stores
        the result. It also recomputes the midplane dust density. If updatestokes
        is False, then the Stokes number is not recomputed, but kept as it is in self.St.
        BEWARE: Make sure that the midplane gas density (self.rhomid) is current.
        NOTE: For computing Stokes number (and stopping time) we keep dv=1e3 for now.
        """
        # nr = len(self.r)
        if updatestokes:
            self.compute_stokes_from_agrain(dv=1e3)
        self.sigma[:] = self.get_dust_radial_drift_next_timestep(dt, alphamodel=alphamodel, fixgas=fixgas, extracond=extracond)
        self.compute_hdust_and_rhodust()

    def get_drift_diffusion_solution(self, Mdot):
        """
        Solve for the steady state of constant mass accretion rate
        taking into account radial drift and radial diffusion.

        The fluxes in the advection diffusion equation are rewritten, and then
        we solve for d/dr (sigmadust) = RHS. This is then integrated outward.
        Inward integration is numerically unstable.

        Definitions:
        ------------

        C = Mdot /(2 * pi * r * D)
        D = diffusion constant
        v = drift speed
        A = v/D + dlog sigma_gas/dr

        Arguments:
        ----------

        Mdot : float
            dust mass accretion rate

        Output:
        -------

        sig_dust : array
            the dust density solution on the grid self.r

        """

        # the definition in the quation is inward accretion = negative mdot
        Mdot = - np.abs(Mdot)

        # note: for odesolve it needs to be y,x for ivp_problem x, y
        def dydx(x, y):
            return dSigdustdr(
                x,
                y,
                self.r,
                self.omk,
                self.diskradialmodel.alpha,
                self.diskradialmodel.cs,
                self.diskradialmodel.hp,
                self.St,
                self.diskradialmodel.sigma,
                Mdot)

        # determine a reasonable inner boundary condition

        gamma = np.log(
            self.diskradialmodel.sigma[1] * self.omk[1] * self.diskradialmodel.cs[1] /
            (self.diskradialmodel.sigma[0] * self.omk[0] * self.diskradialmodel.cs[0])
            ) / np.log(self.r[1] / self.r[0])
        v_in = self.diskradialmodel.cs[0]**2 / (self.r[0] * self.omk[0] * (self.St[0] + 1. / self.St[0])) * gamma
        sig0 = np.abs(Mdot) / np.abs(2 * np.pi * self.r[0] * v_in)

        # call the ode solver

        # sig_dust = odeint(dydx, sig0, self.r)[:, 0]

        sig_dust = solve_ivp(dydx, (self.r[0], self.r[-1]), [sig0], method='BDF', t_eval=self.r).y[0, :]

        return sig_dust

    def join_multi_array(self, componentlist, ir=None):
        """
        If you have a list of disk components, this method creates an
        array of sigmas. If ir is set, then only for ir.
        """
        nc = len(componentlist)
        nr = len(self.r)
        if ir is None:
            sigma = np.zeros((nr, nc))
            for ic in range(nc):
                sigma[:, ic] = componentlist[ic].sigma[:]
        else:
            sigma = np.zeros((nc))
            for ic in range(nc):
                sigma[ic] = componentlist[ic].sigma[ir]
        return sigma

    def return_multi_array(self, componentlist, sigma, ir=None):
        """
        This is the reverse of joint_multi_array: it puts
        the results from an array back into the multiple
        components.
        """
        nc = len(componentlist)
        # nr = len(self.r)
        if ir is None:
            for ic in range(nc):
                componentlist[ic].sigma[:] = sigma[:, ic]
        else:
            for ic in range(nc):
                componentlist[ic].sigma[ir] = sigma[ic]

    def gravinstab_apply_flattening(self):
        """
        If the gas surface density of the disk was 'flattened' using the
        gravinstab_flattening() command (i.e. mass was spead out to guarantee
        Q>=2 everywhere, while keeping mass and angular momentum conserved),
        then the dust must 'move along' with the gas. This is done with 
        this command. 

        WARNING: This function should ONLY be called right after the 
        call of gravinstab_flattening() of the parent radial disk model.
        """
        ifrom = self.diskradialmodel.gravinst_ifrom
        ito   = self.diskradialmodel.gravinst_ito
        frac  = self.diskradialmodel.gravinst_frac
        dsurf = self.diskradialmodel.get_dsurf()
        sigmaorig = self.sigma.copy()
        for i in range(len(ito)):
            dsigma                = sigmaorig[ifrom[i]]*frac[i]
            self.sigma[ito[i]]   += dsigma*(dsurf[ifrom[i]]/dsurf[ito[i]])
            self.sigma[ifrom[i]] -= dsigma
        if(hasattr(self.diskradialmodel,'gravinst_flushfactor_in')):
            self.sigma[0] *= (1.-self.diskradialmodel.gravinst_flushfactor_in)
        if(hasattr(self.diskradialmodel,'gravinst_flushfactor_out')):
            self.sigma[-1] *= (1.-self.diskradialmodel.gravinst_flushfactor_out)


@jit
def dSigdustdr(x, y, r, omk, alpha, cs, hp, St, sigma_g, Mdot):
    """
    the derivative of the dust surface density at radius x in the model

    Arguments:
    ----------

    y : float
        dust surface density

    x : float
        the radius where the derivative is calculated
    """

    # outer boundary condition

    if x > r[-1]:
        x = r[-1]

    # ir is the grid index right of the value x

    ir = r.searchsorted(x)

    # calculate the x-dependent quantities in the left and right cells

    D_R = alpha * cs[ir] * hp[ir] / (1. + St[ir]**2)
    D_L = alpha * cs[ir - 1] * \
        hp[ir - 1] / (1. + St[ir - 1]**2)

    gamma = np.log(
        sigma_g[ir] * omk[ir] * cs[ir] /
        (sigma_g[ir - 1] * omk[ir - 1] * cs[ir - 1])
        ) / \
        np.log(r[ir] / r[ir - 1])

    dsg_dr = (sigma_g[ir] - sigma_g[ir - 1]
              ) / (r[ir] - r[ir - 1])

    if ir == len(r) - 1:
        return y / sigma_g[-1] * dsg_dr

    v_R = cs[ir]**2 / \
        (r[ir] * omk[ir] * (St[ir] + 1. / St[ir])) * gamma
    v_L = cs[ir - 1]**2 / \
        (r[ir - 1] * omk[ir - 1] * (St[ir - 1] + 1. / St[ir - 1])) * gamma

    A_R = v_R / D_R + dsg_dr / sigma_g[ir]
    A_L = v_L / D_L + dsg_dr / sigma_g[ir - 1]

    C_R = Mdot / (2 * np.pi * r[ir]) / D_R
    C_L = Mdot / (2 * np.pi * r[ir - 1]) / D_L

    # interpolation of A and C coefficients

    eps = (x - r[ir - 1]) / (r[ir] - r[ir - 1])

    A = (1 - eps) * A_L + eps * A_R
    C = (1 - eps) * C_L + eps * C_R

    return A * y - C
