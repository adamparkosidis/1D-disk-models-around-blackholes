import numpy as np
from . import natconst as nc
from .diskvertical import DiskVerticalModel
from .grainmodel import GrainModel
import warnings
try:
    from .solvediff3d import solvefld3d, solvediff3d
except ImportError:
    warnings.warn("Importing diskvertical was successful, but 2D radiative diffusion not available, because diffusion.f90 has not been compiled to diffusion.so. All other methods of diskvertical.py remain available. For full functionality type make in disklab directory, which should produce the diffusion.so library using f2py.")
from scipy.interpolate.interpnd import LinearNDInterpolator


class Disk2D(object):
    def __init__(self, disk, zrmax=None, nz=90, meanopacitymodel=None,
                 irradmode='flaring angle', mugas=2.3):
        """
        This object links a series of 1-D vertical disk structure models
        together into a 1+1-D model. It also provides some true 2-D features.

        Arguments:
        ----------
        disk : DiskRadialModel
            A DiskRadialModel object (see diskradial.py) upon which the 2D model is
            derived from.

        Keywords:
        ---------

        zrmax : float | array
            The upper z of the vertical grid in units of r.
            This can be an array (a different zrmax for each r)
            or a scalar (the same zrmax for all r). The latter is
            required if you want to use the 2-D radiative diffusion
            model and/or the radial ray-trace model for irradiation.

        nz : int
            number of vertical grid points

        meanopacitymodel : list
            list of mean opacity parameters, see
            `disklab.meanopacity.evaluate_meanopacity`

        irradmode : str
            Which method to use for calculating the irradiation. Possible
            choices are:
            - 'flaring angle'
            - 'flaring index'
            - 'radial raytrace'
            - 'isothermal': take isothermal temperature based on disk mid-plane
              IMPORTANT: Only if you choose 'isothermal' you will assure that
              the 2-D (or 1+1-D) model will have the same temperature as the
              disk.tmid array (the 1-D radial model).

        mugas : float
            mean molecular weight of the gas in units of proton masses
        """

        # link the parent DiskRadialModel instance and copy the relevant attributes

        self.disk = disk
        if zrmax is None:
            self.zrmax = 6 * (disk.hp / disk.r).max()
        else:
            self.zrmax = zrmax
        nr    = len(disk.r)
        self.nz = nz
        self.r  = disk.r
        self.cyl2d_r  = self.r
        self.cyl2d_rr = np.zeros((nr, nz))
        self.cyl2d_zz = np.zeros((nr, nz))
        mstar = disk.mstar
        lstar = disk.lstar
        alpha = disk.alpha
        Sc    = disk.Sc
        self.mugas = mugas
        self.gamma = 7. / 5.   # NOTE: For cold outer parts this should be 5./3.
        if meanopacitymodel is None and hasattr(disk, 'meanopacitymodel'):
            meanopacitymodel = disk.meanopacitymodel

        # set flaring angle and flaring index depending on irradiation method
        tgas = [None] * len(disk.r)
        if irradmode == 'flaring angle':
            assert hasattr(disk, 'flang'), "Disk radial model object does not contain flaring angle"
            flidx = [None] * len(disk.r)
            if disk.flang is None:
                flang = [None] * len(disk.r)
            else:
                flang = disk.flang * np.ones_like(disk.r)
        elif irradmode == 'flaring index':
            assert hasattr(disk, 'flidx'), "Disk radial model object does not contain flaring index"
            if flidx is None:
                flidx = [None] * len(disk.r)
            else:
                flidx = disk.flidx * np.ones_like(disk.r)
            flang = [None] * len(disk.r)
        elif irradmode == 'radial raytrace':
            flidx = [None] * len(disk.r)
            flang = [None] * len(disk.r)
        elif irradmode == 'isothermal':
            tgas = disk.tmid
            flidx = [None] * len(disk.r)
            flang = [None] * len(disk.r)
        else:
            raise ValueError('Do not know irradmode')

        # set up the nr vertical slices

        self.verts = []
        for ir in range(nr):
            if np.isscalar(self.zrmax):
                zrm = self.zrmax
            else:
                zrm = self.zrmax[ir]

            # create a vertical slice model

            vert = DiskVerticalModel(mstar, disk.r[ir], disk.sigma[ir], nz=nz,
                                     zrmax=zrm, lstar=lstar, flidx=flidx[ir],
                                     flang=flang[ir], meanopacitymodel=meanopacitymodel,
                                     mugas=mugas, gamma=self.gamma, alphavisc=alpha,
                                     Sc=Sc, tgas=tgas[ir], init_opacity=False)

            # if the parent disk has dust, we add a corresponding dust component
            # to the vertical slice as well. We will use the assumption that
            # the dust is well mixed.

            if hasattr(disk, 'dust'):
                for dust in disk.dust:
                    if not hasattr(dust, 'agrain'):
                        dust.compute_agrain_from_stokes()
                    agrain = dust.agrain
                    if not np.isscalar(agrain):
                        agrain = agrain[ir]

                    vert.add_dust(agrain, dtg=dust.sigma[ir] / vert.siggas,
                                  xigrain=dust.xigrain[ir], grainmodel=dust.grain)

            # finally: store the slice in the `verts` attribute

            self.verts.append(vert)

            # And store the z in the self.cyl2d_zz and the r in self.cyl2d_rr

            self.cyl2d_zz[ir, :] = vert.z.copy()
            self.cyl2d_rr[ir, :] = self.r[ir]

        self.cyl2d_zr    = self.cyl2d_zz / self.cyl2d_rr
        self.cyl2d_rpher = np.sqrt(self.cyl2d_rr**2 + self.cyl2d_zz**2)

    def radial_raytrace(self):
        """
        Compute the irradiation of the disk by radial ray-tracing.
        This works only when all the vertical structure models have
        the same z/r grid, i.e. when their z-grids are the same apart
        from the scaling with r.

        Note: Since the vertical structure models are exactly vertical
              (i.e. cylindrical coordinates), we correct for the
              fact that the spherical radius is sqrt(1+z/r) times
              larger than the footpoint radius. This is the geomfact
              correction factor.
        Note: Here the source term due to the irradiation is not,
              as in the 1D vertical model, computed as a flux
              difference, but simply as flux times opacity.
        """
        assert np.isscalar(self.zrmax), "Radial raytracing only works if the z-grids line up, i.e. all have the same z/r"
        assert len(self.verts) == len(self.disk.r), "Radial raytrace only works if you have exactly the same nr of vertical models as radial grid points."
        nr         = len(self.disk.r)
        nz         = self.nz
        r          = self.disk.r[0]
        zr         = self.cyl2d_zz[0, :] / r
        geomfact   = np.sqrt(1.0 + zr**2)
        rspher     = r * geomfact
        irrad_lum  = np.zeros(nz) + self.disk.lstar
        irrad_lum_optthin  = np.zeros(nz) + self.disk.lstar
        self.verts[0].irrad_flux[:-1] = irrad_lum.copy() / (4 * nc.pi * rspher**2)
        self.verts[0].irrad_jmean_optthin = irrad_lum_optthin.copy() / ((4 * nc.pi)**2 * rspher**2)
        for ir in range(1, nr):
            r       = self.disk.r[ir]
            rspher  = r * geomfact
            dr      = (self.disk.r[ir] - self.disk.r[ir - 1]) * geomfact
            rhokap0 = self.verts[ir - 1].rhogas * self.verts[ir - 1].mean_opacity_planck
            rhokap1 = self.verts[ir].rhogas * self.verts[ir].mean_opacity_planck
            rhokap  = 0.5 * (rhokap0 + rhokap1)
            dtau    = rhokap * dr
            irrad_lum[:] *= np.exp(-dtau)
            self.verts[ir].irrad_flux[:-1]  = irrad_lum.copy() / (4 * nc.pi * rspher**2)
            self.verts[ir].irrad_jmean[:] = self.verts[ir].irrad_flux[:-1] / (4 * nc.pi)
            self.verts[ir].irrad_src[:]   = self.verts[ir].irrad_flux[:-1] * rhokap1
            self.verts[ir].irrad_jmean_optthin = irrad_lum_optthin.copy() / ((4 * nc.pi)**2 * rspher**2)

    def compute_dlnpdlnrc(self):
        """
        For the correct computation of the azimuthal velocity v_phi(r,z) we need the
        horizontal radial gradient of the gas pressure. We will here compute the
        right-side gradient (the gradient at index ir will be between ir+1 and ir).
        The very outer radius will contain a copy of the previous.
        """
        nr             = len(self.r)
        self.pgas      = np.zeros_like(self.cyl2d_zz)
        self.pgas_plus = np.zeros_like(self.cyl2d_zz)
        self.dlnpdlnrc = np.zeros_like(self.cyl2d_zz)
        for ir in range(nr):
            self.pgas[ir, :] = self.verts[ir].rhogas * nc.kk * self.verts[ir].tgas / (self.verts[ir].mugas * nc.mp)
        for ir in range(nr - 1):
            self.pgas_plus[ir, :] = np.exp(np.interp(self.cyl2d_zz[ir, :], self.cyl2d_zz[ir + 1, :], np.log(self.pgas[ir + 1, :])))
        for ir in range(nr - 1):
            dlnrc                = np.log(self.r[ir + 1]) - np.log(self.r[ir])
            self.dlnpdlnrc[ir, :] = (np.log(self.pgas_plus[ir, :]) - np.log(self.pgas[ir, :])) / dlnrc
        self.dlnpdlnrc[-1, :] = self.dlnpdlnrc[-2, :]

    def compute_vphi(self, add_selfgrav=False):
        """
        Compute the azimuthal velocity v_phi(r,z) that includes all effects, including
        the pressure gradient.
        """
        nr             = len(self.r)
        self.vphi      = np.zeros_like(self.cyl2d_zz)
        self.cyl2d_vk  = np.zeros_like(self.cyl2d_zz)
        self.compute_dlnpdlnrc()
        for ir in range(nr):
            self.verts[ir].compute_vphi(self.dlnpdlnrc[ir,:],add_selfgrav=add_selfgrav)
            self.vphi[ir,:] = self.verts[ir].vphi.copy()
            self.cyl2d_vk[ir,:] = self.verts[ir].omk_full[:]*self.r[ir]
        self.cyl2d_vphi = self.vphi  # For consistency

    # ------------------- The 2-D cylindrical stuff ---------------------

    def convert_1p1d_to_cyl2d(self, rhogas=False, tgas=False, rhodust=False, irrad=False,
                              visc=False, ross=False):
        """
        In the standard form, the 2-D disk model is, in fact, a series of independent
        1D vertical disk models: self.verts[:].rhogas[:]. However, it is often useful
        to have the data all in one array, e.g. self.cyl2d_rhogas[:,:]. This has to
        be done for each type of data separately, hence the above keyword arguments.
        """
        nr              = len(self.r)
        nz              = len(self.cyl2d_zz[0, :])
        if hasattr(self.disk, 'dust'):
            ndust       = len(self.disk.dust)
        else:
            rhodust = False
        if(rhogas):
            self.cyl2d_rhogas  = np.zeros((nr, nz))
        if(tgas):
            self.cyl2d_tgas    = np.zeros((nr, nz))
        if(rhodust):
            self.cyl2d_rhodust = np.zeros((ndust, nr, nz))
        if(irrad):
            self.cyl2d_irrad_flux  = np.zeros((nr, nz))
            self.cyl2d_irrad_jmean = np.zeros((nr, nz))
            self.cyl2d_irrad_src   = np.zeros((nr, nz))
            self.cyl2d_irrad_jmean_optthin = np.zeros((nr, nz))
        if(visc):
            self.cyl2d_visc_src = np.zeros((nr, nz))
        if(ross):
            self.cyl2d_meanopacross = np.zeros((nr, nz))
        for ir in range(nr):
            if(rhogas):
                self.cyl2d_rhogas[ir, :] = self.verts[ir].rhogas.copy()
            if(tgas):
                self.cyl2d_tgas[ir, :] = self.verts[ir].tgas.copy()
            if(rhodust):
                for idust in range(ndust):
                    self.cyl2d_rhodust[idust, ir, :] = self.verts[ir].dust[idust].rho.copy()
            if(irrad):
                self.cyl2d_irrad_flux[ir, :]  = self.verts[ir].irrad_flux[:-1].copy()
                self.cyl2d_irrad_jmean[ir, :] = self.verts[ir].irrad_jmean.copy()
                self.cyl2d_irrad_src[ir, :]   = self.verts[ir].irrad_src.copy()
                self.cyl2d_irrad_jmean_optthin[ir, :] = self.verts[ir].irrad_jmean_optthin.copy()
            if(visc):
                self.cyl2d_visc_src[ir, :] = self.verts[ir].visc_src.copy()
            if(ross):
                self.cyl2d_meanopacross[ir, :] = self.verts[ir].mean_opacity_rosseland.copy()

    def convert_cyl2d_to_1p1d(self, rhogas=False, tgas=False, rhodust=False, selfgrav=False):
        """
        This is the reverse of convert_1p1d_to_cyl2d().
        """
        nr              = len(self.r)
        # nz              = len(self.cyl2d_zz[0, :])
        if hasattr(self.disk, 'dust'):
            ndust       = len(self.disk.dust)
        for ir in range(nr):
            if(rhogas):
                self.verts[ir].rhogas[:]   = self.cyl2d_rhogas[ir, :]
            if(tgas):
                self.verts[ir].tgas[:]     = self.cyl2d_tgas[ir, :]
            if(rhodust):
                for idust in range(ndust):
                    self.verts[ir].dust[idust].rho[:] = self.cyl2d_rhodust[idust, ir, :]
            if(selfgrav):
                self.verts[ir].selfgrav_fz  = self.cyl2d_selfgrav_fz[ir, :]
                self.verts[ir].selfgrav_fr  = self.cyl2d_selfgrav_fr[ir, :]

    # ------------------- The 2-D spherical stuff ---------------------

    def setup_spherical_coordinate_system(self, nr_thetapadding=20, theta0=0.01, nr_rpadding=None,
                                          thetagrid=None, rout_factor=None, rin_factor=None,
                                          nr_rpadding_in=None):
        """
        In order to be able to transform the coordinates from the 1+1D cylindrical system
        to the 2-D spherical system and back, we have to set up the spherical coordinate
        system first.

        ARGUMENTS:
          nr_thetapadding      Since the z-grid goes only up to zrmax, but the
                               theta-grid must go up to the polar axis, we need
                               some padding in theta. This is the number of grid
                               points used for this padding.
          nr_rpadding          Since the upper-outer edge of the cylindrical grid
                               extends beyond the spherical radial grid if we
                               use (for the spherical grid) the same grid points
                               as for the cylindrical grid, to have a full
                               coverage, we need to padd a few radial grid cells
                               at the outer edge of the spherical grid.
          theta0               The start of the theta grid (very close to theta=0).
          rin_factor           If set, extends the radial grid inward by a factor
                               r_new_inner = rin_factor * r.min()
                               NOTE: If you have rin_factor set, you can set
                                     nr_rpadding_in, separate from nr_rpadding.
          rout_factor          If set, extends the radial grid outward by a factor
                               r_new_outer = rout_factor * r.max(). If not set,
                               but nr_padding is set, then r grid is extended just
                               enough that the cylindrical grid fits inside the
                               spherical grid.
        """
        assert len(self.verts) == len(self.disk.r), "The coordinate transformation only works if you have exactly the same nr of vertical models as radial grid points."
        nz           = len(self.verts[0].z)
        thetaupper   = np.arctan(1. / (self.cyl2d_zz[:, -1] / self.r).max()) * (1.0 - 1e-9)
        if thetagrid is None:
            thetapadding = np.linspace(theta0, thetaupper, nr_thetapadding, endpoint=False)
            thetamain    = np.linspace(thetaupper, nc.pi / 2., nz)
            theta        = np.hstack((thetapadding, thetamain))
        else:
            theta        = thetagrid
        rspher           = self.r
        if(nr_rpadding is not None):
            if rout_factor is not None:
                assert rout_factor > 1, "rout_factor <= 1"
                rexmax   = rspher[-1] * rout_factor
            else:
                rexmax   = rspher[-1] / np.sin(thetaupper)
            rextra       = np.exp(np.linspace(np.log(rspher[-1]), np.log(rexmax), nr_rpadding + 1))
            rextra       = rextra[1:]
            rspher       = np.hstack((rspher, rextra))
            if rin_factor is not None:
                assert rin_factor < 1, "rin_factor >= 1"
                if nr_rpadding_in is None:
                    nr_rpadding_in = nr_rpadding
                rexmin    = rspher.min() * rin_factor
                rextra    = np.exp(np.linspace(np.log(rexmin), np.log(rspher[0]), nr_rpadding_in + 1))
                rextra    = rextra[:-1]
                rspher    = np.hstack((rextra, rspher))
        self.spher_theta  = theta
        self.spher_r      = rspher
        self.spher_phi    = np.array([0.0])
        spher_rr, spher_tt  = np.meshgrid(self.spher_r, self.spher_theta, indexing='ij')
        self.spher_rr     = spher_rr
        self.spher_tt     = spher_tt
        ri                = np.hstack([rspher[0], 0.5 * (rspher[1:] + rspher[:-1]), rspher[-1]])
        ti                = np.hstack([theta[0], 0.5 * (theta[1:] + theta[:-1]), theta[-1]])
        rri, ttc          = np.meshgrid(ri, theta, indexing='ij')
        rrc, tti          = np.meshgrid(rspher, ti, indexing='ij')
        self.spher_vol    = (2.0 * np.pi / 3.0) * (rri[1:, :]**3 - rri[:-1, :]**3) * \
                            (np.cos(tti[:, :-1]) - np.cos(tti[:, 1:]))
        self.spher_rcyl   = self.spher_rr * np.sin(self.spher_tt)
        self.spher_zcyl   = self.spher_rr * np.cos(self.spher_tt)
        self.spher_zrcyl  = self.spher_zcyl / self.spher_rcyl
        self.spher_zrappr = nc.pi / 2 - spher_tt                     # Approximate z/r
        self.spher_zappr  = self.spher_zrappr * self.spher_rr      # Approximate z

    def coordmap_array_2d(self, xx_orig, yy_orig, data_orig, xx_new, yy_new, fill_value=np.nan):
        (nx, ny)   = data_orig.shape
        (nxn, nyn) = xx_new.shape
        points     = np.reshape(np.transpose(np.array((xx_orig, yy_orig)), (1, 2, 0)), (nx * ny, 2))
        ffp        = np.reshape(data_orig, nx * ny)
        q          = LinearNDInterpolator(points, ffp, fill_value=fill_value)
        ffr        = q(xx_new, yy_new).reshape((nxn, nyn))
        return ffr

    def coord_trafo_cyl2d_to_spher2d(self, cyl2d_q, fill_value=np.nan, midplanemargin=False):
        """
        This is a proper coordinate transformation from cylindrical coordinates
        with an r-grid and an r-dependent z-grid to full 2-D spherical coordinates.
        """
        if midplanemargin:
            self.spher_tt[:, -1] = nc.pi / 2 * (1.0 - 1e-9)
        xx_new    = self.spher_rr * np.sin(self.spher_tt)
        yy_new    = self.spher_rr * np.cos(self.spher_tt)
        spher2d_q = self.coordmap_array_2d(self.cyl2d_rr, self.cyl2d_zz, cyl2d_q, xx_new, yy_new, fill_value=fill_value)
        return spher2d_q

    def coord_trafo_spher2d_to_cyl2d(self, spher2d_q, fill_value=np.nan, midplanemargin=False):
        """
        This is the reverse of coord_trafo_cyl2d_to_spher2d()
        """
        if midplanemargin:
            self.spher_tt[:, -1] = nc.pi / 2 * (1.0 + 1e-9)
        xx_old    = self.spher_rr * np.sin(self.spher_tt)
        yy_old    = self.spher_rr * np.cos(self.spher_tt)
        cyl2d_q   = self.coordmap_array_2d(xx_old, yy_old, spher2d_q, self.cyl2d_rr, self.cyl2d_zz, fill_value=fill_value)
        return cyl2d_q

    def solve_2d_rad_diffusion(self, floorparam=1e-6, thist=False, limiter=None,
                               linsol_convcrit=1e-10, linsol_itermax=10000,
                               nonlin_convcrit=1e-3, nonlin_itermax=20,
                               simplecoordtrans=False):
        """
        Solve the 2-D radiative diffusion problem. This function performs the entire
        procedure: starting from the 1+1D model, it converts the data into spherical
        coordinates, then calls the FLD solver, then converts the results back into
        the 1+1D form.

        Note: This method requires the solvediff3d.py library.
        Note: Often the rho*kappa drops so deeply that the diffusion approximation breaks
              down. We must put a lower floor. Given that the dimension of rho*kappa is
              1/cm, the lower limit also must have a dimension of 1/length. We use the
              radius r for that, and set this floor to floorparam/r.
        Note: When you wish to start with a reset flux limiter for the FLD, and you
              have called this method before, you may want to set self.fld_limiter=None,
              because by default the values of the limiter from the earlier call are
              used (for faster convergence).
        Note: If simplecoordtrans==True, then we do not really do a proper coordinate
              transformation from cylindrical to spherical (the latter necessary
              for the radiative diffusion). Instead we simply use the approximation
              that the vertical model is along spherical r_spher=constant lines.
              This makes the mapping from 1+1D to 2-D spherical coordinates trivial.
              But it is, of course, approximate.
        """
        #
        # If either "limiter" is set or is an attribute, then use this array
        # as the flux limiter. Otherwise set fluxlimiter to 1.0 as a start.
        #
        if limiter is None:
            if hasattr(self, 'fld_limiter'):
                limiter = self.fld_limiter
        #
        # Radial grid
        #
        r     = self.disk.r
        #
        # First bring the relevant variables from 1+1D separate vertical
        # structure models to global 2-D cylindrical arrays. This is a trivial
        # operation, simply copying the 1-D vertical model data at each radial
        # point into a 2-D array.
        #
        self.convert_1p1d_to_cyl2d(rhogas=True, tgas=True, rhodust=True, irrad=True,
                                   visc=True, ross=True)
        #
        # Compute some radiative transfer quantities in the cyl2d system
        #
        if not hasattr(self, 'cyl2d_irrad_src'):
            self.cyl2d_irrad_src = 0.
        if not hasattr(self, 'cyl2d_visc_src'):
            self.cyl2d_visc_src = 0.
        self.cyl2d_rt_s     = (self.cyl2d_irrad_src + self.cyl2d_visc_src) / (4 * nc.pi)
        self.cyl2d_rt_a     = self.cyl2d_rhogas * self.cyl2d_meanopacross
        self.cyl2d_rt_aflr  = floorparam / self.cyl2d_rr
        self.cyl2d_rt_t     = self.cyl2d_tgas
        self.cyl2d_rt_jirr  = self.cyl2d_irrad_flux / (4 * nc.pi)
        self.cyl2d_rt_cvrho = self.cyl2d_rhogas * nc.kk / ((self.gamma - 1.0) * self.mugas * nc.mp)
        #
        # Convert from 2D cylindrical to 2D spherical
        #
        if(simplecoordtrans):
            #
            # The transformation here is done in a simplified way, by
            # saying that z = r * (pi/2-theta). So the "z" coordinate
            # is assumed here to be along spheres of constant radial
            # distance to the star. As long as the disk is geometrically
            # thin enough, that is ok.
            #
            self.spher_r        = r
            th                  = np.pi / 2. - self.verts[0].z / self.verts[0].r
            th                  = th[::-1]
            self.spher_theta    = th
            ph                  = np.array([0.0])
            nr                  = len(self.spher_r)
            nth                 = len(self.spher_theta)
            nph                 = 1
            self.spher_rt_a     = np.zeros((nr, nth, nph))
            self.spher_rt_s     = np.zeros((nr, nth, nph))
            self.spher_rt_jirr  = np.zeros((nr, nth, nph))
            self.spher_rt_t     = np.zeros((nr, nth, nph))
            self.spher_rt_cvrho = np.zeros((nr, nth, nph))
            self.spher_rt_aflr  = np.zeros((nr, nth, nph))
            self.spher_rt_a[:, :, 0]      = self.cyl2d_rt_a[:, ::-1]
            self.spher_rt_s[:, :, 0]      = self.cyl2d_rt_s[:, ::-1]
            self.spher_rt_t[:, :, 0]      = self.cyl2d_rt_t[:, ::-1]
            self.spher_rt_jirr[:, :, 0]   = self.cyl2d_rt_jirr[:, ::-1]
            self.spher_rt_cvrho[:, :, 0]  = self.cyl2d_rt_cvrho[:, ::-1]
            self.spher_rt_aflr[:, :, 0]   = self.cyl2d_rt_aflr[:, ::-1]
        else:
            #
            # The transformation here is a proper transformation from cylindrical
            # (with a 1+1D form) to spherical coordinates
            #
            if not hasattr(self, 'spher_rr') or not hasattr(self, 'spher_tt') or \
               not hasattr(self, 'spher_r') or not hasattr(self, 'spher_theta'):
                raise ValueError('Cannot transform to spherical coordinates if spherical grid not yet set.')
            th                  = self.spher_theta
            ph                  = np.array([0.0])
            nr                  = len(self.spher_r)
            nth                 = len(self.spher_theta)
            nph                 = 1
            # nrcyl               = len(self.r)
            # nzcyl               = len(self.cyl2d_zz[0, :])
            self.spher_rt_a     = np.zeros((nr, nth, nph))
            self.spher_rt_s     = np.zeros((nr, nth, nph))
            self.spher_rt_jirr  = np.zeros((nr, nth, nph))
            self.spher_rt_t     = np.zeros((nr, nth, nph))
            self.spher_rt_cvrho = np.zeros((nr, nth, nph))
            self.spher_rt_aflr  = np.zeros((nr, nth, nph))
            self.spher_rt_a[:, :, 0]      = self.coord_trafo_cyl2d_to_spher2d(self.cyl2d_rt_a, fill_value=self.cyl2d_rt_a.min())
            self.spher_rt_s[:, :, 0]      = self.coord_trafo_cyl2d_to_spher2d(self.cyl2d_rt_s, fill_value=0.)
            self.spher_rt_t[:, :, 0]      = self.coord_trafo_cyl2d_to_spher2d(self.cyl2d_rt_t)
            self.spher_rt_jirr[:, :, 0]   = self.coord_trafo_cyl2d_to_spher2d(self.cyl2d_rt_jirr)
            self.spher_rt_cvrho[:, :, 0]  = self.coord_trafo_cyl2d_to_spher2d(self.cyl2d_rt_cvrho, fill_value=self.cyl2d_rt_cvrho.min())
            self.spher_rt_aflr[:, :, 0]   = self.coord_trafo_cyl2d_to_spher2d(self.cyl2d_rt_aflr, fill_value=self.cyl2d_rt_aflr.min())
            #
            # By converting from cylindrical to spherical, you get certain parts in the
            # spherical grid where the interpolation fails because the spherical points
            # are outside the cylindrical grid. For some quantities we can simply set the
            # value to a standard one. For others, we have to do more work. That is done here.
            #
            tirr_optthin   = (self.cyl2d_irrad_jmean_optthin[:, 0] * (nc.pi / nc.ss))**0.25
            for it in range(nth):
                mask = np.where(np.isnan(self.spher_rt_t[:, it, 0]))
                self.spher_rt_t[mask, it, 0]    = tirr_optthin[mask]
                self.spher_rt_jirr[mask, it, 0] = self.cyl2d_irrad_jmean_optthin[mask, 0]
        #
        # To avoid crashing the diffusion algorithm on too low optical depths,
        # add a floor value to the extinction coefficient.
        #
        ii      = np.where(self.spher_rt_a < self.spher_rt_aflr)
        self.spher_rt_a[ii]   = self.spher_rt_aflr[ii]
        #
        # Specify the boundary condition for the Flux-limited diffusion
        #
        bc      = [[['flux', 1., -1., 0.], ['dirichlet', 1e-3, 0., 0.]],
                   [['flux', 1., -1., 0.], ['flux', 1., 0., 0.]],
                   None]
        #
        # Now call the FLD solver for spherical coordinates, and with the irradiation
        # temperature as a source term.
        #
        geom    = 'spherical'
        tirr    = (self.spher_rt_jirr * (nc.pi / nc.ss))**0.25
        t, j, h, l, err = solvefld3d(geom, r, th, ph,
                                     self.spher_rt_a, self.spher_rt_s, bc,
                                     tinit=self.spher_rt_t, tirrad=tirr,
                                     cvrho=self.spher_rt_cvrho,
                                     linsol_convcrit=linsol_convcrit,
                                     linsol_itermax=linsol_itermax, dt=0, retrad=True,
                                     nonlin_convcrit=nonlin_convcrit,
                                     nonlin_itermax=nonlin_itermax,
                                     limiter=limiter, thist=thist)
        #
        # Store into spherical arrays
        #
        if thist:
            self.spher_rt_t = t[-1]
        else:
            self.spher_rt_t = t
        self.spher_rt_jdiff = j
        self.spher_rt_fld_limiter = l
        self.spher_rt_hflux = h
        #
        # Transform some of these quantities back to cylindrical coordinates
        #
        if(simplecoordtrans):
            self.cyl2d_tgas     = self.spher_rt_t[:, ::-1, 0]
            self.cyl2d_rt_t     = self.spher_rt_t[:, ::-1, 0]
            self.cyl2d_rt_jdiff = self.spher_rt_jdiff[:, ::-1, 0]
        else:
            self.cyl2d_tgas     = self.coord_trafo_spher2d_to_cyl2d(self.spher_rt_t[:, :, 0])
            self.cyl2d_rt_t     = self.cyl2d_tgas
            self.cyl2d_rt_jdiff = self.coord_trafo_spher2d_to_cyl2d(self.spher_rt_jdiff[:, :, 0])
        #
        # Re-insert results into the 1+1D vertical structure models
        #
        for ir in range(nr):
            self.verts[ir].tgas   = self.cyl2d_tgas[ir, :]
            self.verts[ir].jdiff  = self.cyl2d_rt_jdiff[ir, :]

    def solve_2d_selfgrav(self, incl_gas=True, incl_dust=True, cellwalls=False,
                          convcrit=1e-10, itermax=10000,
                          simplecoordtrans=False):
        """
        Compute the self-gravitational potential and forces. This function does
        the whole procedure: adding the gas and dust densities, converting them
        to spherical coodinates, running the Poisson solver, converting the potential
        back to cylindrical coordinates, computing the gradients in cylindrical
        coordinates and therewith the body forces. You can choose where these
        bodyforces are computed: at the cell centers (by setting cellwalls=False)
        or at the cell walls (by setting cellwalls=True).

        Note: If simplecoordtrans==True, then we do not really do a proper coordinate
              transformation from cylindrical to spherical (the latter necessary
              for the radiative diffusion). Instead we simply use the approximation
              that the vertical model is along spherical r_spher=constant lines.
              This makes the mapping from 1+1D to 2-D spherical coordinates trivial.
              But it is, of course, approximate.
        """
        #
        # First bring the relevant variables from 1+1D separate vertical
        # structure models to global 2-D cylindrical arrays. This is a trivial
        # operation, simply copying the 1-D vertical model data at each radial
        # point into a 2-D array.
        #
        self.convert_1p1d_to_cyl2d(rhogas=True, rhodust=True)
        #
        # Compute the total density
        #
        self.cyl2d_rho = np.zeros_like(self.cyl2d_rhogas)
        if incl_gas:
            self.cyl2d_rho += self.cyl2d_rhogas
        if incl_dust and hasattr(self, 'cyl2d_rhodust'):
            ndust = len(self.cyl2d_rhodust[:, 0, 0])
            for idust in range(ndust):
                self.cyl2d_rho += self.cyl2d_rhodust[idust, :, :]
        #
        # Convert from 2D cylindrical to 2D spherical
        #
        if(simplecoordtrans):
            #
            # The transformation here is done in a simplified way, by
            # saying that z = r * (pi/2-theta). So the "z" coordinate
            # is assumed here to be along spheres of constant radial
            # distance to the star. As long as the disk is geometrically
            # thin enough, that is ok.
            #
            th                  = np.pi / 2. - self.verts[0].z / self.verts[0].r
            th                  = th[::-1]
            self.spher_theta    = th
            # ph                  = np.array([0.0])
            nr                  = len(self.spher_r)
            nth                 = len(self.spher_theta)
            nph                 = 1
            self.spher_rho      = np.zeros((nr, nth, nph))
            self.spher_rho[:, :, 0] = self.cyl2d_rho[:, ::-1]
        else:
            #
            # The transformation here is a proper transformation from cylindrical
            # (with a 1+1D form) to spherical coordinates
            #
            if not hasattr(self, 'spher_rr') or not hasattr(self, 'spher_tt') or \
               not hasattr(self, 'spher_r') or not hasattr(self, 'spher_theta'):
                raise ValueError('Cannot transform to spherical coordinates if spherical grid not yet set.')
            th                  = self.spher_theta
            # ph                  = np.array([0.0])
            nr                  = len(self.spher_r)
            nth                 = len(self.spher_theta)
            nph                 = 1
            # nrcyl               = len(self.r)
            # nzcyl               = len(self.cyl2d_zz[0, :])
            self.spher_rho      = np.zeros((nr, nth, nph))
            self.spher_rho[:, :, 0] = self.coord_trafo_cyl2d_to_spher2d(self.cyl2d_rho, fill_value=0., midplanemargin=True)
        #
        # Now solve the Poisson equation in spherical coordinates
        #
        self.spher_pot = self.solve_selfgrav_spher(self.spher_rho, convcrit=convcrit, itermax=itermax)
        #
        # Convert back to cylindrical coordinates
        #
        if(simplecoordtrans):
            self.cyl2d_pot     = self.spher_pot[:, ::-1, 0]
        else:
            self.cyl2d_pot     = self.coord_trafo_spher2d_to_cyl2d(self.spher_pot[:, :, 0], midplanemargin=True)
        #
        # Now compute the gradients in cylindrical coordinates
        #
        self.cyl2d_gradpot_r, self.cyl2d_gradpot_z = \
            self.get_gradient_from_potential_cyl_2d(self.cyl2d_pot, cellwalls=cellwalls)
        self.cyl2d_selfgrav_fr, self.cyl2d_selfgrav_fz = \
            -self.cyl2d_gradpot_r, -self.cyl2d_gradpot_z
        #
        # Now compute the gradients in spherical coordinates
        #
        self.spher_gradpot_r, self.spher_gradpot_t = \
            self.get_gradient_from_potential_spher_2d(self.spher_pot, cellwalls=cellwalls)
        self.spher_selfgrav_fr, self.spher_selfgrav_ft = \
            -self.spher_gradpot_r, -self.spher_gradpot_t
        #
        # For completeness, also copy this information into the list of
        # vertical structures
        #
        self.convert_cyl2d_to_1p1d(selfgrav=True)

    def solve_selfgrav_spher(self, rho, convcrit=1e-10, itermax=10000):
        """
        Solve the gravitational Poisson equation in spherical coordinates.

        ARGUMENTS:
          rho             3-D array of mass density (though only the r
                          and theta dimensions are used) in spherical
                          coordinates.

        RETURNS:
          pot             3-D array of the gravitational potential in
                          spherical coordinates.
        """
        diffcoef  = np.ones((len(self.spher_r), len(self.spher_theta), 1))
        source    = 4 * nc.pi * nc.GG * rho
        geom      = 'spherical'
        mass      = (rho[:, :, 0] * self.spher_vol).sum()
        if self.spher_theta.max() < 0.9 * np.pi:
            mass *= 2
        rout      = self.spher_r.max()
        potrout   = -nc.GG * mass / rout
        boundcond = [[['flux', 1, 0, 0], ['dirichlet', -potrout]],
                     [['flux', 1, 0, 0], ['flux', 1, 0, 0]],
                     [['flux', 1, 0, 0], ['flux', 1, 0, 0]]]
        r         = self.spher_r
        theta     = self.spher_theta
        phi       = self.spher_phi
        pot       = -solvediff3d(geom, r, theta, phi, diffcoef, source, boundcond,
                                 convcrit=convcrit, itermax=itermax)
        return pot

    def get_gradient_from_potential_spher_2d(self, pot, cellwalls=False):
        """
        After solving for the potential using solve_selfgrav_spher(), you can now compute
        the actual body force at every location by computing the gradient. Note that this
        is initially computed at the cell walls (because this is the natural location of
        the gradient operator). But (if cellwalls==False) it is then averaged to the cell centers.
        """
        nr        = len(self.spher_r)
        nt        = len(self.spher_theta)
        rr        = self.spher_rr
        tt        = self.spher_tt
        if len(pot.shape) > 2:
            pot2d = pot[:, :, 0]
        else:
            pot2d = pot
        gradient_r  = np.zeros((nr + 1, nt))
        gradient_t  = np.zeros((nr, nt + 1))
        gradient_r[1:-1, :] = (pot2d[1:, :] - pot2d[:-1, :]) / (rr[1:, :] - rr[:-1, :])
        gradient_r[0, :]    = gradient_r[1, :]
        gradient_r[-1, :]   = gradient_r[-2, :]
        rrav               = 0.5 * (rr[:, 1:] + rr[:, :-1])
        gradient_t[:, 1:-1] = (pot2d[:, 1:] - pot2d[:, :-1]) / (tt[:, 1:] - tt[:, :-1]) / rrav
        gradient_t[:, 0]    = gradient_t[:, 1]
        gradient_t[:, -1]   = gradient_t[:, -2]
        if(cellwalls):
            return gradient_r, gradient_t
        else:
            gradient_r = 0.5 * (gradient_r[1:, :] + gradient_r[:-1, :])
            gradient_t = 0.5 * (gradient_t[:, 1:] + gradient_t[:, :-1])
            return gradient_r, gradient_t

    def get_gradient_from_potential_cyl_2d(self, pot, cellwalls=False):
        """
        As get_gradient_from_potential_spher_2d() but now with the potential
        in cylindrical coordinates. Since the z-coordinate grid is, here,
        linearly increasing with r, the radial gradient is not along the
        grid direction. Hence we have to perform, for each radius, an
        interpolation of the potential, to obtain the gradient. For
        simplicity we choose the inner (left) z-grid, and interpolate
        the potential of the outer (right) z-grid onto that. This makes
        a slight error since at the radial interfaces the z-grid is the
        average of the z-grid left and right.
        """
        assert np.isscalar(self.zrmax), "Self gravity only works if the z-grids line up, i.e. all have the same z/r"
        assert len(self.verts) == len(self.disk.r), "Self gravity only works if you have exactly the same nr of vertical models as radial grid points."
        nr        = len(self.cyl2d_rr[:, 0])
        nz        = len(self.cyl2d_zz[0, :])
        rr        = self.cyl2d_rr
        zz        = self.cyl2d_zz
        gradient_r  = np.zeros((nr + 1, nz))
        gradient_z  = np.zeros((nr, nz + 1))
        for ir in range(nr - 2):
            pot_plus           = np.interp(self.cyl2d_zz[ir, :], self.cyl2d_zz[ir + 1, :], pot[ir + 1, :])
            gradient_r[1 + ir, :] = (pot_plus[:] - pot[ir, :]) / (rr[ir + 1, :] - rr[ir, :])
        gradient_r[0, :]    = gradient_r[1, :]
        gradient_r[-1, :]   = gradient_r[-2, :]
        gradient_z[:, 1:-1] = (pot[:, 1:] - pot[:, :-1]) / (zz[:, 1:] - zz[:, :-1])
        gradient_z[:, 0]    = gradient_z[:, 1]
        gradient_z[:, -1]   = gradient_z[:, -2]
        if(cellwalls):
            return gradient_r, gradient_z
        else:
            gradient_r = 0.5 * (gradient_r[1:, :] + gradient_r[:-1, :])
            gradient_z = 0.5 * (gradient_z[:, 1:] + gradient_z[:, :-1])
            return gradient_r, gradient_z

    def laplace_sper_2d(self, pot):
        """
        The 2-D (r,theta) laplace operator, useful for testing the solution of the
        self-gavity solution.
        """
        rr        = self.spher_rr
        tt        = self.spher_tt
        laplace   = np.zeros_like(rr)
        rrl       = 0.5 * (rr[1:-1, 1:-1] + rr[:-2, 1:-1])
        rrr       = 0.5 * (rr[2:, 1:-1]   + rr[1:-1, 1:-1])
        ttl       = 0.5 * (tt[1:-1, 1:-1] + tt[1:-1, :-2])
        ttr       = 0.5 * (tt[1:-1, 2:]   + tt[1:-1, 1:-1])
        sinttl    = np.sin(ttl)
        sinttr    = np.sin(ttr)
        laplace[1:-1, 1:-1] += (1. / rr[1:-1, 1:-1]**2)                                      \
            * ((rrr**2) * (pot[2:, 1:-1] - pot[1:-1, 1:-1]) / (rr[2:, 1:-1] - rr[1:-1, 1:-1]) - # noqa
               (rrl**2) * (pot[1:-1, 1:-1] - pot[:-2, 1:-1]) / (rr[1:-1, 1:-1] - rr[:-2, 1:-1]))  \
            / (rrr - rrl)
        laplace[1:-1, 1:-1] += (1. / (rr[1:-1, 1:-1]**2 * np.sin(tt[1:-1, 1:-1])))              \
            * (sinttr * (pot[1:-1, 2:] - pot[1:-1, 1:-1]) / (tt[1:-1, 2:] - tt[1:-1, 1:-1]) -  # noqa
               sinttl * (pot[1:-1, 1:-1] - pot[1:-1, :-2]) / (tt[1:-1, 1:-1] - tt[1:-1, :-2]))    \
            / (ttr - ttl)
        return laplace


class Disk2DComponent(object):
    """
    To allow the disk to contain one or more dust or chemical components, this class contains
    all the methods needed for moving such a component in 2-D cylindrical coordinates (with
    the r-scaled z grid), including vertical and radial drift and mixing.
    """
    def __init__(self, disk2d, massfraction=None, agrain=None, xigrain=None, grainmodel=None,
                 import_from_verts_ispec=None):
        self.disk2d = disk2d
        if massfraction is not None:
            self.rho    = disk2d.cyl2d_rhogas * massfraction
        else:
            self.rho    = np.zeros_like(disk2d.cyl2d_rhogas)
        if import_from_verts_ispec is not None:
            assert len(disk2d.verts) == self.rho.shape[0], "The nr of 1+1D vertical models is not equal to the nr of radial grid points"
            assert len(disk2d.verts[0].z) == self.rho.shape[1], "The nr of z-grid points in the vertical models is not equal to the nr of z grid points in 2D"
            for ir in len(disk2d.verts):
                self.rho[ir, :] = disk2d.verts[ir].dust[import_from_verts_ispec].rho
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
