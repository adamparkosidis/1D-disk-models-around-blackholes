import numpy as np
from scipy import optimize
from scipy.interpolate import interp1d
from . import natconst as nc
import copy
import warnings
import pkg_resources
import hashlib
import os.path


class GrainModel(object):
    """
    This module contains some subroutines for computing the dynamic properties
    of dust particles in a disk, such as Stokes number and such. Also methods
    for the opacity are included.
    """

    def __init__(self, tstop=None, agrain=None, xigrain=3.0, sublimationmodel=None):
        self.xigrain = xigrain
        if agrain is not None:
            self.agrain = agrain
            self.compute_mgrain()
        if tstop is not None:
            self.tstop = tstop
        self.sublimationmodel = sublimationmodel
        self.species = None

    def compute_mgrain(self):
        """
        Given self.agrain and self.xigrain (the material density), compute the
        mass of the grain.
        """
        self.mgrain  = (4. * nc.pi / 3.) * self.xigrain * (self.agrain**3)

    def compute_tstop(self, rhogas, tgas, dv):
        """
        Wrapper function around tstop_function()
        """
        agrain       = self.agrain
        xigrain      = self.xigrain
        self.tstop   = self.tstop_function(agrain, xigrain, rhogas, tgas, dv)

    def tstop_function(self, agrain, xigrain, rhogas, tgas, dv, retall=False):
        """
        Given agrain [cm] and xigrain [g/cm^3] this routine computes the stopping time
        of the grain. It requires, however, the gas density rhogas [g/cm^3]
        and temperature [K], and possibly also the relative velocity dv [cm/s].
        Returns the stopping time tstop [s].

        References:
           Birnstiel, Dullemond & Brauer (2010) A&A 513, 79.
           Perets & Murray-Clay (2011) ApJ 733, 56.
        """
        absdv        = np.abs(dv)
        sigmah2      = 2e-15                            # Cross section of H_2 - H_2 collisions
        ngas         = rhogas / (2.3 * nc.mp)
        lfree        = 1.0 / (ngas * sigmah2)               # Eq.(9) Birnstiel etal
        cs           = (nc.kk * tgas / (2.3 * nc.mp))**0.5

        # vth=ubar. Note: Eq below Eq.(10) of
        # Birnstiel etal has error. Must be 8/pi.
        # Eq.(8) Birnstiel etal
        # Above Eq.(8) Birnstiel etal
        # Eq.(10) Birnstiel etal

        vth          = cs * (8. / nc.pi)**0.5

        numol        = 0.5 * vth * lfree
        Re           = 2 * agrain * absdv / numol
        if lfree > (4. / 9.) * agrain:
            #
            # Epstein regime
            #
            tstop = (xigrain / rhogas) * (agrain / vth)
        elif Re < 1.:
            #
            # Stokes regime
            #
            tstop = 2 * xigrain * agrain**2 / (9 * numol * rhogas)
        elif Re < 800.:
            #
            # Intermediate Reynolds regime. Here is where various papers
            # deviate.
            #
            # Birnstiel, Dullemond & Brauer (2010) use:
            #
            # tstop = 2.**0.6*xigrain*agrain**1.6 / (9*numol**0.6*rhogas**1.4*absdv**0.4)
            #
            # Perets & Murray-Clay (2011) use (their Eqs. 6 and 7):
            #
            cd    = (24. / Re) * (1. + 0.27 * Re)**0.43 + 0.47 * (1. - np.exp(-0.04 * (Re**0.38)))
            fd    = 0.5 * cd * nc.pi * agrain**2 * rhogas * absdv**2
            mgr   = (4. * nc.pi / 3.) * xigrain * agrain**3
            tstop = mgr * absdv / fd
            #
            # which seems to behave better.
            #
        else:
            #
            # Ram pressure regime
            #
            tstop = 6 * xigrain * agrain / (rhogas * absdv)
        if retall:
            return tstop, Re, lfree, numol
        else:
            return tstop

    def tstop_equation(self, agrain, xigrain, rhogas, tgas, dv, tstop):
        eq = self.tstop_function(agrain, xigrain, rhogas, tgas, dv) - tstop
        return eq

    def solve_agrain(self, rhogas, tgas, dv):
        """
        Given self.tstop (in seconds) this routine tries to find the corresponding
        grain size agrain. It requires, however, the gas density rhogas [g/cm^3]
        and temperature [K], and possibly also the relative velocity dv [cm/s].
        Returns the grain size (radius) agrain in cm. Note that if the resulting
        grain size would be smaller than "alow" or larger than "ahigh", then it
        is limited to those values.
        NOTE: This procedure is not guaranteed to always work correctly, because
              the stopping time is not a monotonous function if agrain in the Stokes
              regime.
        """
        tstop        = self.tstop
        xigrain      = self.xigrain
        alow         = 1e-7  # Minimum grain size in cm
        ahigh        = 1e8   # Maximum grain size in cm
        eqlow        = self.tstop_equation(alow, xigrain, rhogas, tgas, dv, tstop)
        eqhigh       = self.tstop_equation(ahigh, xigrain, rhogas, tgas, dv, tstop)
        if eqlow > 0.:
            self.agrain = alow
        elif eqhigh < 0.:
            self.agrain = ahigh
        else:
            self.agrain = optimize.brentq(self.tstop_equation, alow, ahigh, args=(xigrain, rhogas, tgas, dv, tstop))
        self.compute_mgrain()

    def make_standard_wavelength_grid(self, lammin=1e-1, lammax=1.e5, nlam=120):
        """
        This method creates a standard wavelength grid from 0.1 micron up to 10 cm.
        This is only necessary if an opacity is computed internally (instead of read
        from a file). It may not be the best wavelength grid (in that it may not
        resolve certain dust features). But for the simple opacity model(s) it should
        be fine.
        """
        self.opac_lammic = lammin * (lammax / lammin)**(np.linspace(0., 1., nlam))

    def compute_simple_opacity(self, lam=None, tgrid=None, tabulatemean=True):
        """
        This is a super-simple opacity model (only use for test-calculations!).
        See Ivezic et al. (1997), MNRAS 291, 121.

        ARGUMENTS:
         lam           If given, then compute the opacity only for this/these
                       wavelength(s) (in cm).
                       If not given, then it will create an own wavelength grid.
         tgrid         If given, then for the mean opacity tabulation it will
                       use this temperature grid.
                       If not given, it will create an own temperature grid
                       for the mean opacity table.
         tabulatemean  If True, then tabulate the mean opacity (only possible
                       if the nr of lam wavelengths >4 and the smallest lam
                       is < 2e-5 and the largest lam > 1e-2.
                       If False, then no mean opacity table will be
                       calculated (any old one will remain).
        """
        assert hasattr(self, 'agrain'), 'Cannot compute simple opacity without self.agrain'
        self.compute_mgrain()
        if lam is not None:
            if np.isscalar(lam):
                lam = np.array([lam])
            self.opac_lammic = lam * 1e4     # Convert lam in cm to lammic in micron
        else:
            self.make_standard_wavelength_grid()
            lam = self.opac_lammic / 1e4
        agrain   = self.agrain
        mgrain   = self.mgrain
        xmie     = 2 * nc.pi * agrain / lam
        qabs     = np.ones_like(lam)   # (Approximation for x>1 = big grains or small wavelength)
        ii       = xmie < 1.0
        qabs[ii] = qabs[ii] * xmie[ii]  # (Approximation for x<1 = small grains or large wavelength)
        self.opac_qabs = qabs
        self.opac_kabs = (nc.pi * agrain**2 / mgrain) * qabs
        self.opac_ksca = np.zeros_like(self.opac_kabs)
        self.opac_ksca_eff = self.opac_ksca
        if tabulatemean:
            assert not np.isscalar(lam), "Cannot compute mean opacity for a single given wavelength. Use tabulatemean=False."
            assert lam.min() < 2e-5 and lam.max() > 1e-2, "Cannot compute mean opacity on this wavelength grid"
            self.tabulate_mean_opacities_without_sublimation(tgrid=tgrid)

    def load_standard_opacity(self, reference, species, verbose=False):
        """
        DISKLAB comes with a few precalculated standard dust opacities, which are
        loaded by this method.

        Arguments:
        ----------
        reference : str
            The name of the paper/source where these opacities were taken from.
            NOTE: This is not always the original paper of the lab measurements!

        species : str
            The name of the material / species

        Keywords:
        ---------

        verbose : bool
            If set to True, print statements will tell what is going on.

        IMPORTANT:
        ----------
        When using load_standard_opacity(), the grain size and
        material density will be taken from these standard opacities!
        Any previous value of self.agrain or self.xigrain will be lost!

        CHOICES FOR NOW:
        ----------------
        reference='ddn01':
            The paper Dullemond, Dominik & Natta (2001) ApJ 560, 957.
            These are also the opacities used in the CGPLUS program.
            All these opacities are spherical grains of 0.1 micron radius.
            Available opacities:
            species='silicate': Astronomical (amorphous) silicate from
                                Laor & Draine (1993) ApJ 402, 441
            species='carbon':   Amorphous carbon from
                                Preibisch et al. (1993) A&A 279, 577

        reference='Draine2003':
            Astronomical silicates from [Draine 2003](https://dx.doi.org/10.1086/379123]).
            species='astrosilicates'
        """
        path = pkg_resources.resource_filename(__name__, os.path.join('opacity', 'precalculated'))
        filename = os.path.join(path, 'dustkappa_' + reference + '_' + species + '.npz')

        assert os.path.isfile(filename), 'unknown opacity or file not found: ' + filename

        self.read_opacity(filename, verbose=verbose)

    def read_opacity(self, opacdata, tgrid=None, verbose=False, lammicgrid=None,
                     inherit_density=False, tabulate_mean_opac=True):
        """
        Read opacity from file. Possible formats are:

        - the default disklab opacity format: see `disklab.opacity.write_disklab_opacity`
        - the RADMC-3D opacity format: dustkappa_***.inp.

        This routine will also (as a bonus) compute the tabulated Planck and Rosseland mean
        opacities for this dust opacity. Note that the Rosseland mean opacity of
        a single dust species is only partially meaningful when using a mixture of
        several dust species.

        Arguments:
        ----------

        opacdata : 
          If type(opacdata)==str:
            The file containing the tabulated opacities. The name must end in
            - '.npz' for disklab format, or
            - '.inp' for radmc3d format
          If type(opacdata)==dict or np.lib.npyio.NpzFile
             opacdata should then be a dictionary like the .npz disklab format.

        Keywords:
        ---------

        tgrid : None | array
            If set, then this will be the temperature grid used for the
            tabulated mean opacities.

        verbose : bool
            If set to True, print statements will tell what is going on.

        inherit_density : bool
            If True, then the rho_s (material density) from the opacity
            file (if present) is used as self.xigrain.
            If False, and the material density is not the same as self.xigrain,
            a warning will be given.

        lammicgrid: np.array in units of micron
            If set, this selects these wavelengths (in micron!) from the full
            opacity array. Note: the mean opacities are calculated with the
            original array.

        tabulate_mean_opac: bool
            If True, then precompute and tabulate mean opacities for this
            opacity table.

        """
        #
        # Read the opacity table
        #
        if type(opacdata)==str:
            file_ext = opacdata[-4:]
            assert file_ext in ['.inp', '.npz'], 'Opacity file name must end with .inp (RADMC3D format) or .npz (disklab.opacity format)'
            if verbose:
                print('Reading file ' + opacdata)
        elif type(opacdata)==dict or type(opacdata)==np.lib.npyio.NpzFile:
            file_ext = '.npz'
        else:
            assert 0==1, "Error in read_opacity(): opacdata not understood."

        if file_ext == '.inp':
            with open(opacdata, 'r') as f:
                iformat = int(f.readline())
                if iformat == 2:
                    ncol = 3
                elif iformat == 3:
                    ncol = 4
                else:
                    raise ValueError('Format of opacity file unknown')
                nf      = int(f.readline())
                myarray = np.fromfile(f, dtype=np.float64, count=nf * ncol, sep=' ')
            myarray = myarray.reshape(nf, ncol)
            self.opac_lammic  = 1.0 * myarray.T[0, :]
            self.opac_kabs    = 1.0 * myarray.T[1, :]
            self.opac_ksca    = 1.0 * myarray.T[2, :]
            if iformat == 3:
                self.opac_gsca = 1.0 * myarray.T[3, :]
                self.opac_ksca_eff = self.opac_ksca * (1.0 - self.opac_gsca)  # Ishimaru 1978
            else:
                self.opac_ksca_eff = self.opac_ksca
        elif file_ext == '.npz':
            if type(opacdata)==str:
                data = np.load(opacdata)
            elif type(opacdata)==dict or type(opacdata)==np.lib.npyio.NpzFile:
                data = opacdata
            else:
                assert 0==1, "Error in read_opacity(): opacdata not understood."
            assert 'a' in data, 'opacity dictionary needs to contain particle size array \'a\''
            assert 'lam' in data, 'opacity dictionary needs to contain wave length array \'lam\''
            assert 'k_abs' in data, 'opacity dictionary needs to contain absortpion opacity array \'k_abs\''
            assert 'k_sca' in data, 'opacity dictionary needs to contain scattering opacity array \'k_sca\''
            assert self.agrain is not None, 'for reading opacity from npz-file, the grain size needs to be set'
            a     = data['a']
            lam   = data['lam']
            k_abs = data['k_abs']
            k_sca = data['k_sca']
            assert a[0] <= self.agrain <= a[-1], 'particle size outside of pre-calculated opacity'
            
            if 'rho_s' in data:
                rho_s = float(data['rho_s'])
                if inherit_density:
                    self.xigrain = rho_s
                    self.compute_mgrain()
                    if hasattr(self,'diskradialcomp'):
                        self.diskradialcomp.xigrain[:] = self.xigrain
                        self.diskradialcomp.agrain[:]  = self.agrain
                        self.diskradialcomp.mgrain[:]  = self.mgrain
                else:
                    if np.abs(rho_s / self.xigrain - 1) > 1e-8:
                        warnings.warn('material density of opacity data is {:.2g} does not match density of this grain = {:.2g}'.format(rho_s, self.xigrain))

            # interpolate at the size of the grain
            self.opac_lammic = lam * 1e4

            f = interp1d(np.log10(a), np.log10(k_abs.T))
            self.opac_kabs = 10.**f(np.log10(self.agrain))

            f = interp1d(np.log10(a), np.log10(k_sca.T))
            self.opac_ksca = 10.**f(np.log10(self.agrain))

            # also read and interpolate g if present

            if 'g' in data:
                g = data['g']
                f = interp1d(a, g.T)
                self.opac_gsca = f(self.agrain)
                self.opac_ksca_eff = self.opac_ksca * (1.0 - self.opac_gsca)  # Ishimaru 1978
            else:
                self.opac_ksca_eff = self.opac_ksca
        #
        # Now compute or read the mean opacities
        #
        if tabulate_mean_opac:
            if verbose:
                print('Now computing or reading the mean opacities')
            self.tabulate_mean_opacities_without_sublimation(tgrid=tgrid)
        #
        # If lammicgrid is set, then extract opacities at these wavelengths
        #
        if lammicgrid is not None:
            kabs = np.interp(lammicgrid,self.opac_lammic,self.opac_kabs)
            ksca = np.interp(lammicgrid,self.opac_lammic,self.opac_ksca)
            gsca = np.interp(lammicgrid,self.opac_lammic,self.opac_gsca)
            ksca_eff = np.interp(lammicgrid,self.opac_lammic,self.opac_ksca_eff)
            self.opac_lammic = lammicgrid
            self.opac_kabs   = kabs
            self.opac_ksca   = ksca
            self.opac_gsca   = gsca
            self.opac_ksca_eff = ksca_eff
            
    def read_dustinfo(self, filename, verbose=False):
        """
        If present, parse this file.
        IMPORTANT: If it finds information about the grain size, it will overwrite
        any previous value of self.agrain, and similar for material density self.xigrain!
        """
        if(os.path.isfile(filename)):
            if verbose:
                print('Reading information about this opacity from file ' + filename.strip())
            with open(filename, 'r') as f:
                for line in f:
                    args = line.split()
                    assert args[1].strip() == '=', 'No = detected; cannot interpret file'
                    if args[0].strip() == 'radius_micron':
                        self.agrain = float(args[2].strip()) * 1e-4
                    if args[0].strip() == 'material_density':
                        self.xigrain = float(args[2].strip())
        else:
            if verbose:
                print('No opacity information file ' + filename.strip() + 'found.')

    def plot_opacity(self, abs=True, sca=False):
        """
        Plot the opacity.
        """
        import matplotlib.pyplot as plt
        if abs:
            plt.plot(self.opac_lammic, self.opac_kabs)
        if sca:
            plt.plot(self.opac_lammic, self.opac_ksca)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r'$\lambda [\mu\mathrm{m}]$')
        plt.ylabel(r'$\kappa [\mathrm{cm/g}]$')

    def bplanck(self, freq, temp):
        """
        This function computes the Planck function

                       2 h nu^3 / c^2
           B_nu(T)  = ------------------    [ erg / cm^2 s ster Hz ]
                      exp(h nu / kT) - 1

        Arguments:
             freq  [Hz]            = Frequency in Herz
             temp  [K]             = Temperature in Kelvin
        """
        const1 = nc.hh / nc.kk
        const2 = 2 * nc.hh / nc.cc**2
        const3 = 2 * nc.kk / nc.cc**2
        # const1 = 4.7991598e-11
        # const2 = 1.4745284e-47
        # const3 = 3.0724719e-37
        assert np.isscalar(temp), "Error: bplanck cannot receive a temperature array. Only a scalar allowed."
        if np.isscalar(freq):
            nu = np.array([freq])
        else:
            nu = np.array(freq)
        bpl = np.zeros(len(nu))
        for inu in range(len(nu)):
            x   = const1 * nu[inu] / (temp + 1e-99)
            if x > 1.e-3:
                if x < 300.:
                    bpl[inu] = const2 * (nu[inu]**3) / (np.exp(x) - 1.e0)
                else:
                    bpl[inu] = 0.0
            else:
                bpl[inu] = const3 * (nu[inu]**2) * temp
        if np.isscalar(freq):
            bpl = bpl[0]
        return bpl

    def bplanckdt(self, freq, temp):
        """
        This function computes the temperature derivative of the
        Planck function

             dB_nu(T)     2 h^2 nu^4      exp(h nu / kT)        1
             --------   = ---------- ------------------------  ---
                dT          k c^2    [ exp(h nu / kT) - 1 ]^2  T^2

        Arguments:
             freq  [Hz]            = Frequency in Herz
             temp  [K]             = Temperature in Kelvin
        """
        const1 = nc.hh / nc.kk
        const2 = 2 * nc.hh**2 / (nc.kk * nc.cc**2)
        # const1 = 4.7991598e-11
        # const2 = 7.0764973e-58
        assert np.isscalar(temp), "Error: bplanckdt cannot receive a temperature array. Only a scalar allowed."
        if np.isscalar(freq):
            nu = np.array([freq])
        else:
            nu = np.array(freq)
        bpldt = np.zeros(len(nu))
        for inu in range(len(nu)):
            x   = const1 * nu[inu] / (temp + 1e-290)
            if(x < 300.):
                theexp     = np.exp(x)
                bpldt[inu] = const2 * nu[inu]**4 * theexp / ((theexp - 1.0)**2 * temp**2) + 1.e-290
            else:
                bpldt[inu] = 0.0
        if np.isscalar(freq):
            bpldt = bpldt[0]
        return bpldt

    def solve_dust_temperature_bbstar(self,lstar,tstar,r,convcr=1e-2,nitermax=20,returntgrey=False):
        """
        Solve the equilibrium temperature for this dust grain at a distance r from a
        star with luminosity lstar and blackbody temperature tstar.
        """
        rstar   = np.sqrt(lstar/(4*np.pi*nc.ss*tstar**4))
        tgrey   = np.sqrt(rstar/(2*r))*tstar
        temp    = tgrey
        if returntgrey: return temp
        kapstar = self.planckmean(tstar)
        for iter in range(nitermax):
            kapdust = self.planckmean(temp)
            eps     = kapdust/kapstar
            told    = temp
            temp    = tgrey/eps**0.25
            if(np.abs(told/temp-1.)<convcr):
                return temp
        raise ValueError('No convergence in dust temperature')
    
    def planckmean(self, temp):
        """
        Compute and return the Planck mean absorption opacity.

        ARGUMENTS:
          temp               Temperature of the dust in Kelvin. Can be an array.
        """
        nu   = 1e4 * nc.cc / self.opac_lammic
        dnu  = nu[1:] - nu[:-1]
        if np.isscalar(temp):
            self.kappa_temp   = np.array([temp])
        else:
            self.kappa_temp   = temp
        self.kappa_planck = np.zeros(len(self.kappa_temp))
        for itemp in range(len(self.kappa_temp)):
            bpl  = self.bplanck(nu, self.kappa_temp[itemp])
            up   = (0.5 * (self.opac_kabs[1:] * bpl[1:] + self.opac_kabs[:-1] * bpl[:-1]) * dnu).sum()
            down = (0.5 * (bpl[1:] + bpl[:-1]) * dnu).sum()
            self.kappa_planck[itemp] = up / down
        if np.isscalar(temp):
            self.kappa_temp   = self.kappa_temp[0]
            self.kappa_planck = self.kappa_planck[0]
        return self.kappa_planck

    def rosselandmean(self, temp):
        """
        Compute and return the Rosseland mean total opacity. The scattering opacity is
        reduced by (1-g) according to Ishimaru 1978.

        ARGUMENTS:
          temp               Temperature of the dust in Kelvin. Can be an array.
        """
        nu   = 1e4 * nc.cc / self.opac_lammic
        dnu  = nu[1:] - nu[:-1]
        if np.isscalar(temp):
            self.kappa_temp   = np.array([temp])
        else:
            self.kappa_temp   = temp
        self.kappa_rosseland = np.zeros(len(self.kappa_temp))
        for itemp in range(len(self.kappa_temp)):
            dbpl = self.bplanckdt(nu, self.kappa_temp[itemp])
            up   = (0.5 * (dbpl[1:] + dbpl[:-1]) * dnu).sum()
            down = (0.5 * (dbpl[1:] / (self.opac_kabs[1:] + self.opac_ksca_eff[1:]) + dbpl[:-1] / (self.opac_kabs[:-1] + self.opac_ksca_eff[:-1])) * dnu).sum()
            self.kappa_rosseland[itemp] = up / down
        if np.isscalar(temp):
            self.kappa_temp      = self.kappa_temp[0]
            self.kappa_rosseland = self.kappa_rosseland[0]
        return self.kappa_rosseland

    def tabulate_mean_opacities_without_sublimation(self, tgrid=None):
        """
        Pre-compute the Planck and Rosseland mean opacities for a grid
        of temperatures. The sublimation effects are NOT included here.
        So the mean opacity would still be non-zero even for temperatures
        well above the sublimation temperature. The temperature-dependence
        is ONLY due to the weighting with the Planck function.

        ARGUMENTS:
          tgrid       The temperature grid to use for the precalculating.
                      If None (default) a standard grid is used.
        """
        if tgrid is None:
            temp0 = 1e-1
            temp1 = 1e5
            ntemp = 1000
            tgrid = temp0 * (temp1 / temp0)**(np.linspace(0., 1., ntemp))
        ntemp = len(tgrid)
        self.meanopactable_tgrid           = tgrid.copy()
        self.meanopactable_kappa_planck    = np.zeros_like(self.meanopactable_tgrid)
        self.meanopactable_kappa_rosseland = np.zeros_like(self.meanopactable_tgrid)
        for itemp in range(len(tgrid)):
            temp = tgrid[itemp]
            self.meanopactable_kappa_planck[itemp]    = self.planckmean(temp)
            self.meanopactable_kappa_rosseland[itemp] = self.rosselandmean(temp)

    def write_mean_opacities(self, filename, hashhex):
        """
        Write the tabulated mean opacities to a file.
        """
        with open(filename, 'w') as f:
            ntemp = len(self.meanopactable_tgrid)
            f.write('1\n')     # Format number
            f.write(hashhex + '\n')
            f.write('{0:d}\n'.format(ntemp))
            for itemp in range(ntemp):
                f.write('{0:13.6e}  {1:13.6e}  {2:13.6e}\n'.format(self.meanopactable_tgrid[itemp],
                                                                   self.meanopactable_kappa_planck[itemp],
                                                                   self.meanopactable_kappa_rosseland[itemp]))

    def read_mean_opacities(self, filename):
        """
        Read the tabulated mean opacities to a file.
        """
        with open(filename, 'r') as f:
            ifrm  = int(f.readline().split()[0])
            hsh   = f.readline().split()[0]
            ntemp = int(f.readline().split()[0])
            self.meanopactable_tgrid           = np.zeros(ntemp)
            self.meanopactable_kappa_planck    = np.zeros(ntemp)
            self.meanopactable_kappa_rosseland = np.zeros(ntemp)
            for itemp in range(ntemp):
                line = f.readline().split()
                self.meanopactable_tgrid[itemp]           = float(line[0])
                self.meanopactable_kappa_planck[itemp]    = float(line[1])
                self.meanopactable_kappa_rosseland[itemp] = float(line[2])

    def hash_file_simple(self, filename):
        """
        Return the hash of the file. Simple SHA1 is enough (it is not security relevant).
        """
        with open(filename) as f:
            content = f.read()
            hashhex = hashlib.sha1(content.encode('utf-8')).hexdigest()
        return hashhex

    def read_tabulate_mean_opacities_without_sublimation(self, hashedfile, tabulatedfile, tgrid=None, verbose=False):
        """
        Precomputing and tabulating mean opacities costs computing time.
        Doing this for the same opacity table again and again is a waste of time.
        So this method is the same as tabulate_mean_opacities_without_sublimation(),
        but it will store the results in the file tabulatedfile. If it finds this
        file, it will read it instead of precomputing it again. But what if the
        original opacity file changed? That could be disasterous: it would use the
        old mean opacities. To prevent this, each tabulatedfile starts with a line
        containing the hash of the file hashedfile (which is the original opacity
        file). If hashedfile changes, then its hash will no longer agree with that
        listed in the first line of the tabulatedfile, which will trigger a
        recomputation.
        """
        assert os.path.isfile(hashedfile), "Opacity file does not exist"
        hashhex = self.hash_file_simple(hashedfile)
        computetable = True
        if os.path.isfile(tabulatedfile):
            with open(tabulatedfile) as f:
                iformat  = f.readline()
                prevhash = f.readline().strip()
                if prevhash == hashhex:
                    computetable = False
        if computetable:
            if verbose:
                print('Computing mean opacity table.')
            self.tabulate_mean_opacities_without_sublimation(tgrid=tgrid)
            self.write_mean_opacities(tabulatedfile, hashhex)
        else:
            try:
                if verbose:
                    print('Reading mean opacity table from file ' + tabulatedfile)
                self.read_mean_opacities(tabulatedfile)
            except Exception as e:
                if verbose:
                    print('Tried and failed to read mean opacity file; recomputing it.')
                self.tabulate_mean_opacities_without_sublimation(tgrid=tgrid)
                self.write_mean_opacities(tabulatedfile, hashhex)

    def get_planckmean_tabulated_without_sublimation(self, temp):
        """
        Find the Planck mean opacity from the tabulated values that are
        precalculated using the tabulate_mean_opacities_without_sublimation()
        method.

        ARGUMENTS:
          temp           Temperature in Kelvin (can be an array)
        """
        assert hasattr(self, 'meanopactable_kappa_planck'), 'Sorry, cannot use the precalculated opacity table if it has not been calculated yet.'
        kappa = np.interp(temp, self.meanopactable_tgrid, self.meanopactable_kappa_planck)
        return kappa

    def get_rosselandmean_tabulated_without_sublimation(self, temp):
        """
        Find the Rosseland mean opacity from the tabulated values that are
        precalculated using the tabulate_mean_opacities_without_sublimation()
        method.

        ARGUMENTS:
          temp           Temperature in Kelvin (can be an array)
        """
        assert hasattr(self, 'meanopactable_kappa_rosseland'), 'Sorry, cannot use the precalculated opacity table if it has not been calculated yet.'
        kappa = np.interp(temp, self.meanopactable_tgrid, self.meanopactable_kappa_rosseland)
        return kappa

    def get_sublimation_parameters(self, species=None):
        """
        For some species the phase transition parameters are given here.
        Included species are 'H2O', 'NH3', 'CO2', 'H2S', 'C2H6', 'CH4', 'CO',
        'MgFeSiO4', 'Mg2SiO4', 'MgSiO3', 'Fe', 'Al2O3', 'SiO2', 'FeS'
        More may follow.

        Data for the ices come from Okuzumi et al. 2016, the formula being

          ln(p_eq) = -A/T + B

        Data for silicates come from Kama et al. 2014, the formula being

          ln(rho_eq) = -A/T + B - ln(T)
        """
        if species is None:
            species = self.species
        param = {}
        if species == 'H2O':
            # Bauer et al. 1997
            param['peq_a'] = 6070.
            param['peq_b'] = 30.86
            param['mu']    = 2 + 16
        elif species == 'CO':
            # Yamamoto et al. 1983; Okuzumi et al. 2016
            param['peq_a'] = 981.8
            param['peq_b'] = 26.41
            param['mu']    = 12 + 16
        elif species == 'NH3':
            # Yamamoto et al. 1983; Okuzumi et al. 2016
            param['peq_a'] = 3754.
            param['peq_b'] = 30.21
            param['mu']    = 3 + 14
        elif species == 'CO2':
            param['peq_a'] = 3148.
            param['peq_b'] = 30.01
            param['mu']    = 12 + 16 * 2
        elif species == 'H2S':
            # Haynes 2014; Okuzumi et al. 2016
            param['peq_a'] = 2860.
            param['peq_b'] = 27.70
            param['mu']    = 3 + 32
        elif species == 'C2H6':
            # Moses et al. 1992; Okuzumi et al. 2016
            param['peq_a'] = 2498.
            param['peq_b'] = 30.24
            param['mu']    = 12 * 2 + 6
        elif species == 'CH4':
            # Yamamoto et al. 1983; Okuzumi et al. 2016
            param['peq_a'] = 1190.
            param['peq_b'] = 24.81
            param['mu']    = 12 + 4
        elif species == 'MgFeSiO4':
            # Pollack et al. 1994; Kama et al. 2009
            param['rhoeq_a'] = 28030.
            param['rhoeq_b'] = 12.471
            param['mu']      = 24  # Estimate; not important since mu drops out
        elif species == 'Mg2SiO4':
            # Cameron & Fegley 1982; Kama et al. 2009
            param['rhoeq_a'] = 26091.
            param['rhoeq_b'] = 13.418
            param['mu']      = 24  # Estimate; not important since mu drops out
        elif species == 'Fe':
            # Pollack et al. 1994; Kama et al. 2009
            param['rhoeq_a'] = 21542.
            param['rhoeq_b'] = 6.6715
            param['mu']      = 56  # Estimate; not important since mu drops out
        elif species == 'Al2O3':
            # Cameron & Fegley 1982; Kama et al. 2009
            param['rhoeq_a'] = 40720.
            param['rhoeq_b'] = 18.479
            param['mu']      = 24  # Estimate; not important since mu drops out
        elif species == 'MgSiO3':
            # Pollack et al. 1994; Kama et al. 2009
            param['rhoeq_a'] = 30478.
            param['rhoeq_b'] = 14.898
            param['mu']      = 24  # Estimate; not important since mu drops out
        elif species == 'SiO2':
            # Schick 1960, Lamy 1974; Kama et al. 2009
            param['rhoeq_a'] = 26335.
            param['rhoeq_b'] = 11.184
            param['mu']      = 28  # Estimate; not important since mu drops out
        elif species == 'FeS':
            # Pollack et al. 1994; Kama et al. 2009
            param['rhoeq_a'] = 155.91
            param['rhoeq_b'] = -4.9516
            param['mu']      = 56  # Estimate; not important since mu drops out
        return param

    def abundance_after_sublimation(self, abun0, rhogas, temp, sublimationmodel=None):
        """
        Compute if, and by how much, the abundance of this dust grain species is
        reduced (or even completely put to zero) by the effect of thermal sublimation
        (often called "evaporation"). If there is no reduction at all (low temperature)
        then this function returns the same value as the input abundance (abun0).

        ARGUMENTS:
          abun0                  The initial dust abundance (by weight) of this grain species.
                                 This would be the value if nothing has sublimated.
          rhogas                 The density of the gas in g/cm^3. This means that
                                 pgas = rhogas*temp, and rhograin = rhogas*abun0.
          temp                   Temperature in Kelvin
          sublimationmodel       This is a list. Its first element is a string with
                                 the name of the sublimation model. The next
                                 elements are parameters of this model. If not set,
                                 then the self.sublimationmodel is used.

        SUBLIMATION MODELS (possible values of sublimationmodel[0]):
          'tsub'         Sublimation at a given temperature. The parameters of this
                         model are given as a dictionary: param = sublimationmodel[1].
                         The possible values of param are:
                           param['tsub']  = The sublimation temperature in K (mandatory parameter)
                           param['plaw']  = For T>tsub the abundance is not immediately
                                            set to 0, but reduced according to a powerlaw
                                            with temperature (must be a negative number!).
                                            (optional parameter).
          'peq'          Equilibrium vapor pressure model. The parameters of this
                         model are given as a dictionary: param = sublimationmodel[1].
                           param['peq_a'] = The parameter a in K
                           param['peq_b'] = The dimensionless parameter b
                           param['mu']    = The molecular weight of the vapor particles
                         The formula for the equilibrium vapor pressure in dyne/cm^2 is then:
                            p^eq = exp( - a/T + b )
                         Alternative to param['a'] and param['b'] you can also specify a
                         standard species name:
                           param['species'] = String that can have one of names recognized
                                              by the get_sublimation_parameters() function.
                                              See that function for more details.

        RETURNS:
          abundance      This is the abundance after the sublimation has taken effect.
                         If no sublimation happened, it is the same as abun0.

        EXAMPLE:
          from disklab.grainmodel import *
          from disklab.natconst import *
          import matplotlib.pyplot as plt
          import copy

          rhogas = 1e-10
          abun0  = 1e-3
          t0     = 100.
          t1     = 2200.
          temp   = t0 * (t1/t0)**np.linspace(0,1,220)

          grain  = GrainModel()
          grain.sublimationmodel = ['peq',{'species':'H2O'}]
          grain.abundance_after_sublimation(abun0,rhogas,temp)
          abun_full = grain.abundance.copy()
          grain.sublimationmodel = ['tsubfrompeq',{'species':'H2O','plaw':-10}]
          grain.abundance_after_sublimation(abun0,rhogas,temp)
          abun_simple = grain.abundance.copy()

          plt.figure()
          plt.plot(temp,abun_full)
          plt.plot(temp,abun_simple)
          plt.xlabel('T [K]')
          plt.ylabel('Abundance')
          plt.xscale('log')
          plt.yscale('log')
          plt.show()

          grain  = GrainModel()
          grain.sublimationmodel = ['peq',{'species':'Mg2SiO4'}]
          grain.abundance_after_sublimation(abun0,rhogas,temp)
          abun_full = grain.abundance.copy()
          grain.sublimationmodel = ['tsubfrompeq',{'species':'Mg2SiO4','plaw':-10}]
          grain.abundance_after_sublimation(abun0,rhogas,temp)
          abun_simple = grain.abundance.copy()

          plt.plot(temp,abun_full)
          plt.plot(temp,abun_simple)
          plt.show()
        """
        if sublimationmodel is None:
            sublimationmodel = self.sublimationmodel
        if np.isscalar(temp):
            self.subl_temp   = np.array([temp])
        else:
            self.subl_temp   = temp
        stemp     = self.subl_temp
        abundance = np.zeros_like(self.subl_temp)
        if sublimationmodel is None:
            assert hasattr(self, 'sublimationmodel'), 'Need sublimationmodel.'
            sublimationmodel = self.sublimationmodel
        if sublimationmodel is not None:
            if sublimationmodel[0] == 'tsub':
                #
                # The simplest model: just a sublimation temperature and an abundance slope.
                #
                tsub = sublimationmodel[1]['tsub']
                if hasattr(sublimationmodel[1], 'plaw'):
                    plaw = sublimationmodel[1]['plaw']
                    assert plaw <= 0.0, 'Powerlaw for sublimation abundance reduction must be negative or zero.'
                else:
                    plaw = None
                abundance[:] = abun0
                ii = (stemp >= tsub)
                if plaw is None:
                    abundance[ii] = 0.0
                else:
                    abundance[ii] = abundance[ii] * (stemp[ii] / tsub)**plaw
            elif sublimationmodel[0] == 'peq':
                #
                # The equilibrium vapor pressure model
                #
                if 'species' in sublimationmodel[1]:
                    #
                    # A selected set of built-in dust species. This can be
                    # extended to any kind of species.
                    #
                    species = sublimationmodel[1]['species']
                    param = self.get_sublimation_parameters(species)
                    mu    = param['mu']
                    if 'peq_a' in param:
                        a     = param['peq_a']
                        b     = param['peq_b']
                        peq   = np.exp(-a / stemp + b)
                    elif 'rhoeq_a' in param:
                        a     = param['rhoeq_a']
                        b     = param['rhoeq_b']
                        rhoeq = np.exp(-a / stemp + b) / stemp
                        peq   = rhoeq * nc.kk * stemp / (mu * nc.mp)
                    else:
                        raise ValueError('parameter of sublimation model unreadable.')
                else:
                    #
                    # User-specified phase curve.
                    #
                    a   = sublimationmodel[1]['peq_a']
                    b   = sublimationmodel[1]['peq_b']
                    mu  = sublimationmodel[1]['mu']
                    peq = np.exp(-a / stemp + b)
                #
                # Store
                #
                self.peq = peq
                self.mu  = mu
                #
                # Now compute the abundance of the remaining solid state volatile
                #
                pvapmax = abun0 * rhogas * nc.kk * stemp / (mu * nc.mp)
                abundance[:] = 0.0
                ii = (pvapmax >= peq)
                abundance[ii] = (1.0 - peq[ii] / pvapmax[ii]) * abun0
            elif sublimationmodel[0] == 'tsubfrompeq':
                #
                # Using the equilibrium vapor pressure model, find the temperature
                # where all the solids are sublimated. Then make a simple dummy
                # model of the abundance reduction such that there is a smooth
                # phase transition. This can be necessary for radiation hydro,
                # for instance, where too-sudden jumps in the opacity can lead to
                # numerical problems. The famous Bell & Lin opacity is such a
                # model.
                #
                if 'species' in sublimationmodel[1]:
                    #
                    # A selected set of built-in dust species. This can be
                    # extended to any kind of species.
                    #
                    species = sublimationmodel[1]['species']
                    param = self.get_sublimation_parameters(species)
                    mu = param['mu']
                    if 'peq_a' in param:
                        pvapmax = abun0 * rhogas * nc.kk * stemp / (mu * nc.mp)
                        a     = param['peq_a']
                        b     = param['peq_b']
                        tsub  = a / (b - np.log(pvapmax))
                    elif 'rhoeq_a' in param:
                        a     = param['rhoeq_a']
                        b     = param['rhoeq_b']
                        rho0  = abun0 * rhogas
                        tsub  = 1000. + np.zeros_like(temp)
                        for iter in range(4):  # Is this enough?
                            tsub = a / (b - np.log(tsub * rho0))
                    else:
                        raise ValueError('parameter of sublimation model unreadable.')
                else:
                    #
                    # User-specified phase curve.
                    #
                    a       = sublimationmodel[1]['peq_a']
                    b       = sublimationmodel[1]['peq_b']
                    mu      = sublimationmodel[1]['mu']
                    pvapmax = abun0 * rhogas * nc.kk * stemp / (mu * nc.mp)
                    tsub    = a / (b - np.log(pvapmax))
                #
                # Now implement the sublimation in the same way as model 'tsub' (see above)
                #
                if 'plaw' in sublimationmodel[1]:
                    plaw = sublimationmodel[1]['plaw']
                    assert plaw <= 0.0, 'Powerlaw for sublimation abundance reduction must be negative or zero.'
                else:
                    plaw = None
                abundance[:] = abun0
                ii = (stemp >= tsub)
                if plaw is None:
                    abundance[ii] = 0.0
                else:
                    abundance[ii] = abundance[ii] * (stemp[ii] / tsub[ii])**plaw
            else:
                raise ValueError('Sublimation model unknown')
        else:
            #
            # No sublimation model given, so keep the abundance as-is
            #
            abundance = abun0
        #
        # Now store and return
        #
        if not np.isscalar(abundance):
            if len(abundance) == 1:
                abundance = abundance[0]
        self.abundance = abundance
        return abundance


class mixedgrains(object):
    """
    Mixedgrains is a class that allows you to combine any number of grainmodels together,
    each with a certain abundance, and compute average properties of the ensemble.

    ARGUMENTS:
      grainlist          List of grainmodel objects

    OPTIONAL ARGUMENTS:
      wavelenmicron      Wavelength grid in micron used to put all opacity tables onto a
                         common wavelength grid. If not specified, then all wavelength grids
                         must be identical.
      abundancelist      List or array of abundances of these grains
      temp               Temperature of the dust in Kelvin
      trad               Radiation temperature in Kelvin (if not specified, it is assumed equal to temp)
    """

    def __init__(self, grainlist, wavelenmicron=None, abundancelist=None, temp=None, trad=None):
        self.grainlist     = grainlist
        for grain in grainlist:
            assert hasattr(grain, 'opac_lammic'), "Please specify wavelength-dependent opacities for each GrainModel object."
        if wavelenmicron is not None:
            self.wavelenmicron = copy.copy(wavelenmicron)
        else:
            self.wavelenmicron = copy.copy(grainlist[0].opac_lammic)
            for grain in grainlist:
                assert len(grain.opac_lammic) == len(self.wavelenmicron), "If wavelenmicron is not given, then the wavelength arrays of all grains must be identical."
                assert (np.abs(grain.opac_lammic / self.wavelenmicron - 1.0)).max() < 1e-6, "If wavelenmicron is not given, then the wavelength arrays of all grains must be identical."
        if abundancelist is not None:
            self.abundancelist = abundancelist
            self.compute_mixed_opacity(abundancelist, temp=temp, trad=trad)

    def compute_mixed_opacity(self, abundancelist, temp=None, trad=None, rhogas=None, autosublim=False):
        """
        Compute, from the list of grains and the corresponding abundances, an average grain model
        with suitably averaged opacities.

        ARGUMENT:
          abundancelist         A list of mass-weighted abundances of the grains in the self.grainlist
          temp                  Temperature of the dust in Kelvin
          trad                  Radiation temperature in Kelvin (if not specified, it is assumed equal to temp)
          rhogas                The gas density, necessary when including dust sublimation
          autosublim            If True, then include sublimation physics (if present in the grain model)
                                to reduce the dust abundances. Default=False.

        RESULT (general):
          self.avgrain          It will produce a GrainModel object, self.avgrain, which contains
                                the wavelength-dependent opacity table of the mixture. This is
                                computed by simple addition, with the abundancelist as the
                                weights. If you specify 'temp', then it will also compute the
                                Planck mean and Rosseland mean opacity.
          self.opac_lammic      Wavelength grid (in micron) of the mixed opacity
          self.opac_kabs        Absorption opacity of the mixed opacity as a function of wavelength
          self.opac_ksca        Scattering opacity of the mixed opacity as a function of wavelength
          self.opac_ksca_eff    Scattering opacity, but not reduced to account for the forward scattering

        RESULTS (if 'temp' is specified):
          self.kappa_temp       The temperature(s) of the following mean opacities
          self.kappa_planck     Planck mean opacity (link to self.avgrain.kappa_planck)
          self.kappa_rosseland  Rosseland mean opacity (link to self.avgrain.kappa_planck)

        ADDITIONAL INPUT FOR SUBLIMATION PHYSICS:
          If sublimationmodel is set in one or more of the grain models (see above what
          shape sublimationmodel should have), AND the autosublim==True, then the abundances in
          the abundancelist will be adjusted for the effect of sublimation. See the function
          abundance_after_sublimation().
        """
        if trad is None:
            if temp is not None:
                trad = temp
        if temp is not None:
            assert np.isscalar(temp), 'Sorry, for mixed opacities the temperature must be a scalar.'
        self.abundancelist = abundancelist
        assert len(self.grainlist) == len(abundancelist), "The grainlist does not have the same number of elements as the abundancelist."
        nspec = len(self.grainlist)
        interpolate = True
        for grain in self.grainlist:
            if len(grain.opac_lammic) == len(self.wavelenmicron):
                if (np.abs(grain.opac_lammic / self.wavelenmicron - 1.0)).max() < 1e-6:
                    interpolate = False
        abun_after_sublim = copy.deepcopy(abundancelist)
        if autosublim:
            if temp is not None:
                for ispec in range(nspec):
                    species = self.grainlist[ispec]
                    if hasattr(species, 'sublimationmodel'):
                        assert rhogas is not None, 'For computing sublimation, the rhogas is needed'
                        abun_after_sublim[ispec] = species.abundance_after_sublimation(abun_after_sublim[ispec],
                                                                                       rhogas, temp,
                                                                                       sublimationmodel=species.sublimationmodel)
        avgrain = GrainModel()
        self.avgrain = avgrain
        avgrain.opac_lammic   = self.wavelenmicron
        avgrain.opac_kabs     = np.zeros_like(avgrain.opac_lammic)
        avgrain.opac_ksca     = np.zeros_like(avgrain.opac_lammic)
        avgrain.opac_ksca_eff = np.zeros_like(avgrain.opac_lammic)
        if not interpolate:
            for ispec in range(nspec):
                grain = self.grainlist[ispec]
                avgrain.opac_kabs     += grain.opac_kabs * abun_after_sublim[ispec]
                avgrain.opac_ksca     += grain.opac_ksca * abun_after_sublim[ispec]
                avgrain.opac_ksca_eff += grain.opac_ksca_eff * abun_after_sublim[ispec]
        else:
            for ispec in range(nspec):
                grain = self.grainlist[ispec]
                avgrain.opac_kabs     += np.interp(self.wavelenmicron, grain.opac_lammic, grain.opac_kabs) * abun_after_sublim[ispec]
                avgrain.opac_ksca     += np.interp(self.wavelenmicron, grain.opac_lammic, grain.opac_ksca) * abun_after_sublim[ispec]
                avgrain.opac_ksca_eff += np.interp(self.wavelenmicron, grain.opac_lammic, grain.opac_ksca_eff) * abun_after_sublim[ispec]
        #
        # Now make links to the avgrain opacities, for easier access
        #
        self.opac_lammic   = avgrain.opac_lammic
        self.opac_kabs     = avgrain.opac_kabs
        self.opac_ksca     = avgrain.opac_ksca
        self.opac_ksca_eff = avgrain.opac_ksca_eff
        if trad is not None:
            avgrain.planckmean(trad)     # Planck mean can be two-temperature (dust temp and radiation temp)
            avgrain.rosselandmean(temp)  # Rosseland mean must always be single-temperature
            self.kappa_temp      = avgrain.kappa_temp
            self.kappa_planck    = avgrain.kappa_planck
            self.kappa_rosseland = avgrain.kappa_rosseland
        else:
            self.kappa_temp      = None
            self.kappa_planck    = None
            self.kappa_rosseland = None
