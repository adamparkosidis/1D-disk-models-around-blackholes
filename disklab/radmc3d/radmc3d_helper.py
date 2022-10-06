import numpy as np
from .. import natconst as nc
import matplotlib.pyplot as plt
import subprocess
import os
import warnings

from disklab.natconst import au, pc


def bplanck(freq, temp):
    """
    This function computes the Planck function

                   2 h nu^3 / c^2
       B_nu(T)  = ------------------    [ erg / cm^2 s ster Hz ]
                  exp(h nu / kT) - 1

    Arguments:
    ----------

    freq : float
        Frequency [Hz]

    temp : float
        temperature [K]

    Output:
    -------

    bpl: float
        Planck function [ erg / (cm^2 s ster Hz)]
    """

    h = nc.hh
    k = nc.kk
    cc = nc.cc

    const1 = h / k
    const2 = 2 * h / cc**2
    const3 = 2 * k / cc**2
    x = const1 * freq / (temp + 1e-99)
    if x > 1.e-3:
        bpl = const2 * (freq**3) / (np.exp(x) - 1e0)
    else:
        bpl = const3 * (freq**2) * temp
    return bpl


def write(fid, *args, **kwargs):
    """
    Helper function to write out automatically formated
    values separated by spaces in one line.

    Arguments:
    ----------
    fid : file handle
        file handle of the file into which to write

    *args : list of arguments
        every argument is written to the file, separated by `sep`
        which is usually a single space

    Keywords:
    ---------
    fmt : format string
        usually empty, and will automatically format elements to string
        can be set to any python format statement (whatever is between
        the {}-brackets).

    sep : str
        separating string. can be new line, then every argument is
        written in new line.

    Examples:
    ---------

    Write a "1" in a single line:
    >>> write(fid,1)

    Write the 1D array `A` to the file, one element per line
    >>> write(fid,*A,sep='\n')
    """
    fmt = kwargs.pop('fmt', '')
    sep = kwargs.pop('sep', ' ')
    fid.write(sep.join([('{' + fmt + '}').format(a) for a in args]) + '\n')


def radmc3d(command, executable='radmc3d', path=os.curdir):
    """
    convenience function to run RADMC3D with the given commands
    and write out the result in real time

    Arguments:
    ----------
    command : str
        radmc3d command, such as 'image inc 45':

    Kwargs:
        executable : string
        default is 'radmc3d' but a full path to radmc3d can be given such as
        './radmc3d' for a locally compiled version

    path : str
        directory in which run radmc3d
    """
    proc = subprocess.Popen([executable] + command.split(),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            cwd=path)
    for line in iter(proc.stdout.readline, b''):
        print('>>>   ' + line.decode().rstrip())


def plotgrid(r=None, ri=None, theta=None, thetai=None, data=None, xlim=None, ylim=None, zlim=None,
             zax=None, ax=None, ptype='contourf', style='xy', nlevels=None, **kwargs):
    """
    Plot a quantity on the grid. The quantity is assumed to be in logspace, otherwise you need to
    specify zlim and nlevels accordingly

    Keywords:
    ---------

    r : array
        the radial grid centers (1D)

    ri : array
        the radial grid interfaces (1D)

    theta : array
        theta grid centers (1D)

    thetai : array
        theta grid interfaces (1D)

    data : array
         2D data array

    xlim : array (2 elements)
        the axes extend in y

    ylim : array (2 elements)
        the axes extend in y

    zlim : array (2 elements)
        the axes extend in z

    zax : array
        specifies the ticks (and extend) of the z-axis. Usually
        this only needs to be set for linear z data, for logarithmic
        zdata, its usually easier to specify zlim and possibly nlevels.

    ax : [*None* | matplotlib.axes]
        into which axes to plot the figure, if *None*, new figure is created

    ptype : string
        specifies what kind of plot to produce. Options are
        -    contourf
        -    contour
        -    pcolormesh

    style : string
        specifies which way the axes are formatted
        -    'xy': normal 2D spacial plot
        -    'rt': r-theta plot

    nlevels : int
        number of coutour levels (for contour plots)

    **kwargs will be passed to the plotting command. For an example, see below-

    Example:
    --------

    >>> plotgrid(r=ri/AU,theta=thetai,data=log10(rhodust_t),ptype='pcolormesh',**{'edgecolor':'face','alpha':0.5})
    >>> gca().set_axis_bgcolor('k')
    >>> draw()

    """
    #
    # assing values if they are not set
    #
    if zlim is None:
        zlim = np.ceil(data[np.invert(np.isnan(data))].max()) + np.array([-20, 0])
    if nlevels is None:
        nlevels = int(zlim[-1] - zlim[0] + 1)
    if zax is None:
        zax = np.linspace(zlim[0], zlim[-1], nlevels)
    if ax is None:
        f, ax = plt.subplots()
    else:
        f = plt.figure(ax.figure.number)
        f.sca(ax)
    #
    # calculate X and Y coordinates from r and theta
    #
    if style == 'xy':
        X = np.tile(r[:, None], [1, len(theta)]) * \
            np.sin(np.tile(theta, [len(r), 1]))
        Y = np.tile(r[:, None], [1, len(theta)]) * \
            np.cos(np.tile(theta, [len(r), 1]))
        Xi = np.tile(ri[:, None], [1, len(thetai)]) * \
            np.sin(np.tile(thetai, [len(ri), 1]))
        Yi = np.tile(ri[:, None], [1, len(thetai)]) * \
            np.cos(np.tile(thetai, [len(ri), 1]))
        if xlim is None:
            xlim = [0, Xi.max()]
        if ylim is None:
            ylim = [0, Yi.max()]
    elif style == 'rt':
        X = r
        Xi = ri
        Y = np.pi / 2 - theta
        Yi = np.pi / 2 - thetai
        data = data.T
        if xlim is None:
            xlim = [0, Xi.max()]
        if ylim is None:
            ylim = [0, Yi.max()]
    #
    # plotting
    #
    if ptype == 'contourf':
        ax.contourf(X, Y, data, zax, **kwargs)
    elif ptype == 'contour':
        ax.contour(X, Y, data, zax, **kwargs)
    elif ptype == 'pcolormesh':
        ax.pcolormesh(Xi, Yi, data, vmin=zax[0], vmax=zax[-1], clim=[zax[0], zax[-1]], **kwargs)
    #
    # plot formatting
    #
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    return ax


def plotmesh(ri=None, thetai=None, r=None, theta=None, style='xy', scale=None, ax=None):
    """
    Plot the grid mesh

    Keywords:
    ---------

    ri : float
        radial grid interfaces

    thetai : float
        theta grid interfaces

    r : float
        radial grid center positions

    theta : float
        theta grid center positions

    scale : float
        divide distance units by this factor, default: AU

    style : string
         'xy' = axes are x and y
         'rt' = axes are r and theta
    """
    scale = scale or au
    if ax is None:
        fs = max(plt.rcParams['figure.figsize'])
        f = plt.figure(figsize=[fs, fs])
        ax = f.add_subplot(111)
        if style == 'xy':
            ax.set_xlabel('x')
            ax.set_ylabel('y')
        elif style == 'rt':
            ax.set_xlabel('r')
            ax.set_ylabel(r'$\pi/2-\theta$')
    #
    # plot radial grid centers
    #
    if style == 'xy':
        if r is not None and theta is not None:
            r = r / au
            for t in theta:
                ax.plot(r * np.sin(t), r * np.cos(t),
                        c='k', ls='', marker='.')
        if ri is not None and thetai is not None:
            ri = ri / au
            for t in thetai:
                ax.plot(ri * np.sin(t), ri * np.cos(t), c='k', ls='-')
            for rr in ri:
                ax.plot(rr * np.sin(thetai), rr * np.cos(thetai), c='k', ls='-')
    if style == 'rt':
        if r is not None and theta is not None:
            r = r / au
            for t in theta:
                ax.plot(r, np.pi / 2 - t * np.ones(len(r)), c='k', ls='', marker='.')
        if ri is not None and thetai is not None:
            ri = ri / au
            for t in thetai:
                ax.plot(ri, np.pi / 2 - t * np.ones(len(ri)), c='k', ls='-')
            for rr in ri:
                ax.plot(rr * np.ones(len(thetai)), np.pi / 2 - thetai, c='k', ls='-')


def plot_vert_structure(disk2d, kappa_dust=1e2, tcontours=True):
    """
    Plot the density, temperature, and radial optical depth for a 2d model.

    Arguments:
    ----------
    disk2d : instance of disklab.Diskvert2d
        grids and quantities are derived from this object

    Keywords:
    ---------
    kappa_dust : float | array
        absorption opacity assumed when calculating dust densities [cm^2/g]
        needs to be a float or an array with the shape (nr, ntheta, ndust)

    tcontours : bool
        whether or not to plot contour lines of the temperature

    Output:
    -------
    X,Z : arrays
        radius and vertical height grids

    rho, T, tau_r : arrays
        density, temperature, and optical depth arrays on grid X, Z

    f : figure
        the figure handle of the plot
    """
    from scipy.integrate import cumtrapz
    # Get grid arrays from verts

    Z = np.array([vert.z for vert in disk2d.verts])
    X = disk2d.r[:, None] * np.ones_like(Z)

    # Get arrays for $\rho$, $T$, and $\tau$

    rho = np.array([[dust.rho for dust in vert.dust] for vert in disk2d.verts]).swapaxes(1, 2)
    rho_t = rho.sum(-1)
    T = np.array([vert.tgas for vert in disk2d.verts])
    tau_r = cumtrapz((rho * kappa_dust).sum(-1).T, x=disk2d.r, initial=1e-100).T

    # PLOTTING

    f, axs = plt.subplots(3, 1, figsize=(15, 8), sharex=True, sharey=True)

    # density

    vmax = np.ceil(np.log10(rho_t.max()))
    cc = axs[0].pcolormesh(X / au, Z / X, np.log10(rho_t + 1e-100),
                           vmin=vmax - 8, vmax=vmax)  # gist_heat_r
    cb = plt.colorbar(cc, ax=axs[0])
    cb.set_label(r'$\log(\rho)$ [g cm$^{-3}$]')

    # temperature

    vmax = np.ceil(T.max())
    cc = axs[1].pcolormesh(X / au, Z / X, T, vmin=1,
                           vmax=vmax, cmap='coolwarm')  # gist_heat_r
    cb = plt.colorbar(cc, ax=axs[1])
    cb.set_label(r'$T$ [K]')

    c = axs[1].contour(X / au, Z / X, T, 10, colors='w')
    plt.clabel(c, inline=1, fontsize='xx-small')

    # radial optical depth

    cc = axs[2].pcolormesh(
        X / au, Z / X, np.log10(tau_r + 1e-100), vmin=-1, vmax=1, cmap='RdGy')
    cb = plt.colorbar(cc, ax=axs[2])
    cb.set_label(r'$\log(\tau_r)$ ')

    # labels

    for ax in axs:
        ax.set_ylabel('$z/r$')
    ax.set_xlabel('$r$ [au]')
    ax.set_ylim(0, 0.4)
    ax.set_xscale('log')
    f.subplots_adjust(hspace=0.1, wspace=0.05)

    return X, Z, rho, T, tau_r, f


def write_stars_input(d, lam_mic, path=os.curdir):
    """
    Writes out the stars.inp file based on the stellar properties of the
    disklab model d. For now just using a blackbody star.

    Arguments:
    ----------

    d : model instance from disklab
        takes stellar properties from this model

    lam_mic : array
        wavelength array in micrometers

    path : str
        directory into which to write the file
    """
    with open(os.path.join(path, 'stars.inp'), 'w') as f:
        write(f, 2)  # format identifier
        write(f, 1, len(lam_mic))  # number of stars, number of wavelengths
        write(f, d.rstar, d.mstar, 0, 0, 0)  # radius, mass, xyz position
        write(f, *lam_mic, sep='\n')         # the wavelength grid

        # if only a negative value is given, then this is used as Teff of a BB spectrum

        write(f, -d.tstar)


def write_grid(ri, thetai, phii, mirror=True, path=os.curdir):
    """
    Writes the grid to the RADMC3D inputfile amr_grid.inp

    Arguments:
    ----------
    ri : array
        radial interface grid

    thetai : array
        theta grid, measuring angle up from mid-plane.

    phii : array
        phi grid, should be [0,2*np.pi] for disklab

    mirror : bool
        by default (True) mirror symmetry is used, so only the top half of the disk
        is passed to RADMC3D. If `mirror=False`, the grid is extended to cover
        the lower side of the disk as well.

    path : str
        directory into which to write the file
    """
    if mirror and (thetai[0] != 0):
        raise AssertionError("thetai should start at 0 and increase - will be transformed to radmc3d coordinates")

    if not np.isclose(phii[0], 0.0):
        raise AssertionError("phii[0] should be zero")
    if not np.isclose(phii[-1], 2 * np.pi):
        raise AssertionError("phii[1] should be 2*pi")

    with open(os.path.join(path, 'amr_grid.inp'), 'w') as f:
        write(f, 1)        # format identifier
        write(f, 0)        # 0 = regular grid
        write(f, 100)      # 100 = spherical
        write(f, 0)        # (always 0)
        write(f, 1, 1, int(len(phii) > 2))  # include x- and y-coordinate
        if mirror:
            write(f, len(ri) - 1, len(thetai) - 1, len(phii) - 1)  # grid size
        else:
            write(f, len(ri) - 1, 2 * (len(thetai) - 1), len(phii) - 1)  # grid size

        # the cell interface radii

        write(f, *ri, sep='\n')

        # the cell interface theta

        if mirror:
            # just convert to theta=0 being the pole
            write(f, *(np.pi / 2 - thetai[::-1]), sep='\n')
        else:
            # convert to theta=0 being the pole and add the other side of disk

            write(f, *(np.hstack((
                np.pi / 2 - thetai[::-1],
                np.pi / 2 + thetai[1:]
            ))), sep='\n')

        # the cell interfaces in phi

        write(f, *phii, sep='\n')


def write_dust_density(rho2, mirror=True, fname='dust_density.inp', path=os.curdir):
    """
    Write the dust density array rho2 to the radmc3d input file dust_density.inp

    Arguments:
    ----------
    rho2 : np.array
        2D array (nr, ntheta) or 3D array (nr, ntheta, nspecies), where the first theta
        value corresponds to mid-plane value if mirroring is used.

    mirror : bool
        if `mirror=False` also the lower half of the disk is written out.

    fname : str
        file name into which to write

    path : str
        directory into which to write the file
    """
    if rho2.ndim == 2:
        rho2 = rho2[:, :, None]

    if not mirror:
        rho2 = np.hstack((rho2[:, ::-1, :], rho2))

    n_species = rho2.shape[-1]

    with open(os.path.join(path, fname), 'w') as f:
        write(f, 1)                    # Format identifier
        write(f, np.prod(rho2.shape[:-1]))  # Number of cells
        write(f, n_species)  # Nr of dust species

        for i_spec in range(n_species):

            # write out the density

            write(f, *(rho2[:, ::-1, i_spec].ravel(order='F')), sep='\n')

        write(f, '\n')


def write_density3d(rho3, fname='dust_density.inp', path=os.curdir):
    """
    Write a 3d density array rho3 to the radmc3d input file dust_density.inp

    Arguments:
    ----------
    rho3 : np.array
        3D array (nr, ntheta, nphi) where the first theta
        value corresponds to the lowest theta angle where theta is measured
        from the pole down

    fname : str
        filename to write to. defaults to dust_density.inp

    path : str
        directory into which to write the file
    """

    with open(os.path.join(path, fname), 'w') as f:
        write(f, 1)                    # Format identifier
        write(f, np.prod(rho3.shape))  # Number of cells
        write(f, 1)  # Nr of dust species
        write(f, *(rho3[:, :, :].ravel(order='F')), sep='\n')
        write(f, '\n')


def write_wavelength_micron(lam_mic, path=os.curdir):
    """
    writes the wavelength grid (given in micron) to the RADMC3D input file wavelength_micron.inp

    Arguments:
    ----------
    lam_mic : array
        wavelength grid in micron

    path : str
        directory into which to write the file
    """
    with open(os.path.join(path, 'wavelength_micron.inp'), 'w') as f:
        write(f, len(lam_mic))  # length
        write(f, *lam_mic, sep='\n')


def write_opacity_grid(lam_mic, kappa_abs, kappa_sca=None, gfact=None, name='silicate', path=os.curdir):
    """
    Writes out the opacity information of a grain species for RADMC3D.

    Arguments:
    ----------
    lam_mic : array
        wavelength grid in micron

    kappa_abs : array
        absorption opacity on wavelength grid lam_mic [cm^2/g]

    Keywords:
    ---------
    kappa_sca : None | array
        scattering opacity on wavelength grid lam_mic [cm^2/g]

    gfact : array
        Henyey-Greenstein factor on grid lam_mic

    name : str
        name to be appended to

    path : str
        directory into which to write the file
    """
    nlam = len(lam_mic)

    data = np.vstack((lam_mic, kappa_abs))

    if kappa_sca is not None:
        data = np.vstack((data, kappa_sca))
    if gfact is not None:
        data = np.vstack((data, gfact))

    with open(os.path.join(path, 'dustkappa_{}.inp'.format(name)), 'w') as f:
        if gfact is not None:
            write(f, 3)  # format
        else:
            write(f, 2)  # format
        write(f, nlam)  # length
        for row in data.T:
            write(f, *row)


def write_opacity(disk2d, path=os.curdir, key=None):
    """
    writes the RADMC-3D opacity files (dustkappa_[name].inp) and the opacity
    info (dustopac.inp) based on the given disk2d model.

    path : str
        directory into which to write the file
    """

    if key is None:
        def key(grain):
            return grain.agrain

    # get unique list of dust species and their IDs

    species = list(set([dust.grain for vert in disk2d.verts for dust in vert.dust]))
    species = sorted(species, key=key)
    id_species = [id(spec) for spec in species]

    # for each species: write the opacity data to file

    for i_spec, spec in enumerate(species):

        write_opacity_grid(
            spec.opac_lammic,
            spec.opac_kabs,
            kappa_sca=spec.opac_ksca,
            gfact=spec.opac_gsca, name=str(id_species[i_spec]),
            path=path)

    # write out the names for which we use the IDs for now

    write_opacity_info(species=[str(id) for id in id_species], path=path)


def write_opacity_info(species=['silicate'], path=os.curdir):
    """
    Write out the opacity info file for RADMC3D.

    Keywords:
    ---------
    species : array
        For now, just the names and thermal grains are always assumed.

    path : str
        directory into which to write the file
    """
    with open(os.path.join(path, 'dustopac.inp'), 'w') as f:
        write(f, '2               Format number of this file')
        write(f, '{}              Nr of dust species'.format(len(species)))

        for name in species:
            write(f, '============================================================================')
            write(f, '1               Way in which this dust species is read')
            write(f, '0               0=Thermal grain')
            write(f, '{}              Extension of name of dustkappa_***.inp file'.format(name))

        write(f, '----------------------------------------------------------------------------')


def write_radmc3d_input(params={}, path=os.curdir):
    """
    Write the RADMC3D input file radmc3d.inp. This contains some defaults, but
    whatever is given as key, value pair in the dictionary `params` will overwrite
    those defaults.

    Keywrods:
    ---------
    params : dict
        parameters and values written to radmc3d.inp.

    path : str
        directory into which to write the file
    """
    defaults = {
        'nphot': int(1e6),
        'istar_sphere': 1,
        'scattering_mode_max': 1,
        'scattering_mode': 1,
        'modified_random_walk': 1
    }

    # write possible input to the defaults

    for k, v in params.items():
        defaults[k] = v

    # write parameters to file

    maxlength = np.max([len(k) for k in defaults.keys()])
    with open(os.path.join(path, 'radmc3d.inp'), 'w') as f:
        for k, v in defaults.items():
            write(f, k.ljust(maxlength) + ' = ' + str(v))


def get_radmc3d_arrays(disk2d, nr_add=10, dr_add=0.05, showplots=False, key=None):
    """
    Reads the vertical structure from the Diskvert2d object and creates a 2D grid
    that is refined at the boundary and hopefully ready to be written out for a RADMC3D run.

    Arguments:
    ----------
    disk2d : disklab.diskvertical.diskvert2d
        vertical structure model from disklab

    Keywords:
    ---------
    nr_add : int
        nr of additional grid points

    dr_add: float
        fraction of inner grid point by which to extend the grid inwards

    key : None | callable
        key function to sort the list of species (and therefore densities)
        if None, defaults to `lambda grain: grain.agrain`
        else, need to return something to be used as sorting key in `sorted`

    showplots : bool
        if turned on, will produce lots of debugging plots
    """
    from scipy.interpolate import griddata
    from scipy.integrate import cumtrapz

    if key is None:
        def key(grain):
            return grain.agrain

    # GET DATA FROM OBJECT

    # get peak wavelength

    lam_peak = 0.29 / disk2d.disk.tstar

    # cylindrical radius

    x = disk2d.r.copy()

    # Get grid arrays from verts

    Z0 = np.array([vert.z for vert in disk2d.verts])
    X0 = disk2d.r[:, None] * np.ones_like(Z0)

    # get a list of unique species, then sort them

    species = list(set([dust.grain for vert in disk2d.verts for dust in vert.dust]))
    species = sorted(species, key=key)

    id_species = [id(spec) for spec in species]
    n_species = len(species)

    # for each vert and each dust, we check which species in our species list
    # we are using, and then we assign it to the appropriate array

    rho0 = np.zeros([len(disk2d.r), disk2d.nz, n_species])
    temp = np.zeros([len(disk2d.r), disk2d.nz])
    for ir, vert in enumerate(disk2d.verts):
        temp[ir, :] = vert.tgas
        for dust in vert.dust:
            i_spec = id_species.index(id(dust.grain))
            rho0[ir, :, i_spec] = dust.rho

    # we also create the opacity array at a given wavelength
    kappa_dust = np.array([np.interp(lam_peak, spec.opac_lammic * 1e-4, spec.opac_ksca) for spec in species])[None, None, :]

    # CREATE SPHERICAL GRID
    # create spherical radius cell centers and interfaces from cylindrical radius
    # we take the interfaces at the centers between the current cells, but then
    # recalculate the cell centers again, extrapolating the boundary interfaces

    ri1 = 0.5 * (x[1:] + x[:-1])
    ri1 = np.hstack((ri1[0] - (ri1[2] - ri1[1]), ri1,
                     ri1[-1] * ri1[-2] / ri1[-3]))
    r1 = 0.5 * (ri1[1:] + ri1[:-1])

    # Create a $\theta$-grid, just the mid-plane density value needs to be
    # adjusted - we know the mid-plane density, but we need to interpolate to
    # get the first grid point which has it's lower interface at $z=0$.

    # note: this is defined from the mid-plane up

    theta = np.arctan(disk2d.cyl2d_zr[0])
    thetai = 0.5 * (theta[1:] + theta[:-1])
    thetai = np.hstack((0, thetai, theta[-1] + (theta[-1] - thetai[-1])))
    theta = 0.5 * (thetai[1:] + thetai[:-1])

    # now create intermediate X and Y grids. We will refine those to smooth the inner edge

    X2_c = r1[:, None] * np.cos(theta[None, :])
    Y2_c = r1[:, None] * np.sin(theta[None, :])
    X2_i = ri1[:, None] * np.cos(thetai[None, :])
    # Y2_i = ri1[:, None] * np.sin(thetai[None, :]) # not used

    # next, since we updated the cell centers, we will interpolate the old density structure onto this new grid
    rho_sph = np.zeros([X2_c.shape[0], X2_c.shape[1], n_species])
    points = np.array(list(zip(X0.flatten(), Z0.flatten())))
    newpoints = np.array(list(zip(X2_c.flatten(), Y2_c.flatten())))
    for idust in range(n_species):
        values = np.log10(rho0[:, :, idust] + 1e-100).flatten()
        rho_sph[:, :, idust] = 10.**griddata(
            points, values, newpoints, fill_value=-100).reshape(len(r1), len(theta))

    # also interpolate the temperature

    values = np.log10(temp[:, :] + 1e-100).flatten()
    temp_sph = 10.**griddata(
        points, values, newpoints, fill_value=-100).reshape(len(r1), len(theta))

    # as sanity checks, we can plot the previous (cylindrical) and new (spherical) grids

    if showplots:
        f, axs = plt.subplots(2, 1, figsize=(8, 8), sharex=True, sharey=True)
        rho0_tot = rho0.sum(-1)
        vmax = np.ceil(np.log10(rho0_tot.max()))
        axs[0].pcolormesh(X0 / au, Z0 / X0, np.log10(rho0_tot + 1e-100),
                          vmin=vmax - 8, vmax=vmax, edgecolor=(1, 1, 1, 0.5), linewidth=0.1)
        axs[1].pcolormesh(X2_c / au, Y2_c / X2_c, np.log10(rho_sph.sum(-1) + 1e-100),
                          vmin=vmax - 8, vmax=vmax, edgecolor=(1, 1, 1, 0.5), linewidth=0.1)
        axs[0].set_xscale('log')
        axs[0].set_ylim(0, 0.3)
        axs[0].set_xlim(0.99 * X2_i[0, 0] / au, X2_i[20, 0] / au)

    # get kappa for each species at a given wavelength

    tau1 = cumtrapz((rho0 * kappa_dust).sum(-1).T, x=r1, initial=1e-100, axis=1).T

    # Smooth inner edge
    # based on rho0, the opacity, and the spherical grid, we
    # add a smoothed-off region to refine the transition from
    # optically thick to thin

    ref = refine_inner_edge(r1, ri1, rho_sph, temp, kappa_dust, nr_add=nr_add, showplots=showplots)
    rho2 = ref['rho']
    temp2 = ref['temp']
    tau2 = ref['tau']
    r2 = ref['r']
    ri2 = ref['ri']

    if showplots:

        # Plot the new density and optical depth at the mid-plane for testing

        f, ax = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
        ax[0].semilogy(r1 / au, tau1[:, 0], '-x')
        ax[0].semilogy(r2 / au, tau2[:, 0], '--+')
        ax[0].set_ylim(1e-7, 1e3)
        ax[0].set_ylabel(r'$\tau$')

        ax[1].loglog(r1 / au, rho0_tot[:, 0], '-x')
        ax[1].loglog(r2 / au, rho2.sum(-1)[:, 0], '--+')
        ax[1].set_ylim(1e-10 * rho0_tot.max(), 2 * rho0_tot.max())
        ax[1].set_ylabel(r'$\rho$')

        ax[1].set_xlim(0.9 * disk2d.r[0] / au, disk2d.r[30] / au)

    # Reassign the default names

    r = r2
    ri = ri2
    rho = rho2
    rho2_tot = rho2.sum(-1)

    nphi = 1
    nr = len(r)
    nth = len(theta)

    phii = np.linspace(0, 2 * np.pi, nphi + 1)

    X_c = r[:, None] * np.cos(theta[None, :])
    Y_c = r[:, None] * np.sin(theta[None, :])
    X_i = ri[:, None] * np.cos(thetai[None, :])
    Y_i = ri[:, None] * np.sin(thetai[None, :])

    if showplots:

        xlim = [0.99 * ri[0] / au, 1.01 * ri[12] / au]
        ylim = np.array([0.0, 0.25])

        # Plot the $\tau_r$ distribution before and after adding a smoothed edge

        vmax = np.ceil(np.log10(rho2_tot.max()))
        plotgrid(
            r=r / au, ri=ri / au, theta=np.pi / 2 - theta,
            thetai=np.pi / 2 - thetai, data=np.log10(rho2_tot + 1e-100),
            xlim=xlim, ylim=ylim * X_c[0, 0] / au, zlim=[vmax - 7, vmax],
            ptype='pcolormesh', linewidth=0.1, edgecolor='face', alpha=0.5)
        ax = plt.gca()
        ax.set_facecolor('0.5')
        ax.add_artist(plt.Circle((0, 0), radius=disk2d.disk.rstar / au, ec='w', fc='y'))
        ax.set_aspect('equal')
        ax.set_title('new density grid')

        plotmesh(ri=ri, thetai=np.pi / 2 - thetai, r=r, theta=np.pi / 2 - theta, scale=au)
        ax = plt.gca()
        ax.axhline(0, c='r')
        ax.add_artist(plt.Circle((0, 0), radius=disk2d.disk.rstar / au, ec='w', fc='y'))
        ax.set_xlim(xlim)
        ax.set_ylim(ylim * X_c[0, 0] / au)
        ax.set_aspect('equal')
        ax.set_title('new grid')

        f, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True, sharey=True)
        cc1 = axs[0].pcolormesh(
            X0 / au, Z0 / X0, np.log10(tau1 + 1e-100),
            vmin=-1, vmax=1, cmap='RdGy', edgecolor=(1, 1, 1, 0.5), lw=0.1)
        cc2 = axs[1].pcolormesh(
            X_c / au, Y_c / X_c, np.log10(tau2 + 1e-100),
            vmin=-1, vmax=1, cmap='RdGy', edgecolor=(1, 1, 1, 0.5), lw=0.1)
        for ax in axs:
            ax.set_ylim(ylim)
            ax.set_xlim(xlim)
            ax.set_ylabel('$z/r$')

        ax.set_xlabel('$r$ [au]')

        cb1 = plt.colorbar(cc1, ax=axs[0])
        cb2 = plt.colorbar(cc2, ax=axs[1])
        cb1.set_label(r'$\log(\tau_r)$ ')
        cb2.set_label(r'$\log(\tau_r)$ ')
        f.subplots_adjust(hspace=0.05, wspace=0.05)

        f, ax = plt.subplots(1, 1, figsize=(10, 4))
        cc = ax.pcolormesh(
            X_c / au, Y_c / X_c, temp2,
            vmin=100, vmax=1500, cmap='coolwarm', edgecolor=(1, 1, 1, 0.5), lw=0.1)
        ax.set_ylim(ylim)
        ax.set_xlim(xlim)
        ax.set_ylabel('$z/r$')
        ax.set_xlabel('$r$ [au]')

        cb = plt.colorbar(cc, ax=ax)
        cb.set_label(r'$T$ ')
        f.subplots_adjust(hspace=0.05, wspace=0.05)

    # return the new grid and density structure

    return {
        'r': r,
        'ri': ri,
        'theta': theta,
        'thetai': thetai,
        'phii': phii,
        'xc': X_c,
        'xi': X_i,
        'yc': Y_c,
        'yi': Y_i,
        'rho': rho,
        'species': species,
        'temp': temp2,
        'nr': nr,
        'nphi': nphi,
        'nth': nth
    }


def refine_inner_edge(r, ri, rho, temp, kappa_dust, nr_add=10, showplots=False):
    """
    Takes a spherical 2D grid and adds a softened refined inner edge to
    trace the transition from optically thick to thin.

    r : array
        spherical radius grid: cell center

    ri : array
        spherical radius grid: cell interfaces

    rho : array
        spherical density grid (nr, ntheta, ndust), itheta=0 is mid-plane

    temp : array
        spherical gas temperature

    kappa_dust : float | array
        dust opacity as cm^2/g

    nr_add : int
        number of additional inner grid cells for refinement

    showplots : bool
        whether to show debuggin plots

    Output:
    -------
    dictionary containing the following items

    r : array
        new cell center grid, len(r) = nr = (nr_initial + nr_add)

    ri : array
        new cell interface grid, len(ri) = nr + 1

    rho : array
        new density grid, shape(ri) = [nr, ntheta, ndust]

    tau : array
        new radial optical depth grid, shape(ri) = [nr, ntheta]
    """
    from scipy.integrate import cumtrapz
    # calculate tau_r at the mid-plane

    tau1 = cumtrapz((rho * kappa_dust).sum(-1).T, x=r, initial=1e-100, axis=1).T

    # Add grid points and a steeply declining density extrapolation to smooth the transition

    i_ref = tau1[:, 0].searchsorted(1)  # inside of which cell to refine

    if i_ref < len(r):

        # calculate a fine linear grid (interfaces and centers) inside the inner interface

        ri_add = np.linspace(ri[i_ref] * (1 - 0.05), ri[i_ref], nr_add + i_ref)
        r_add = 0.5 * (ri_add[1:] + ri_add[:-1])

        # add a radially exponentially dropping density

        rho_add = np.exp(
            (r_add - r_add[-1]) / (r_add[-1] - r_add[-2]))[:, None, None] * rho[i_ref, :, :]

        temp_add = np.ones_like(r_add)[:, None] * temp[i_ref, :]

        # create the extended grids, that is, i.e. fine linear grid + old grid

        r2 = np.hstack((r_add, r[i_ref:]))
        ri2 = np.hstack((ri_add, ri[i_ref + 1:]))
        # rho2 = np.vstack((rho_add, rho[i_ref:, :]))
        rho2 = np.concatenate((rho_add, rho[i_ref:, ...]), axis=0)
        temp2 = np.concatenate((temp_add, temp[i_ref:, ...]), axis=0)

        # calculate the radial optical depth on that grid
        tau2 = cumtrapz((rho2 * kappa_dust).sum(-1).T, x=r2, initial=1e-100, axis=1).T

        if showplots:

            xlim = [0.99 * ri2[0] / au, 1.01 * ri2[nr_add + 2] / au]

            # show the old grid, the linear extension grid,
            # and the new combined grid

            f, ax = plt.subplots()
            ax.plot(r / au, 1.2 * np.ones_like(r), 'o', c='C0')
            ax.plot(ri / au, 1.2 * np.ones_like(ri), '+', c='C0')
            ax.plot(r_add / au, 1.0 * np.ones_like(r_add), 'o', c='C1')
            ax.plot(ri_add / au, 1.0 * np.ones_like(ri_add), '+', c='C1')
            ax.plot(r2 / au, 0.8 * np.ones_like(r2), 'o', c='C2')
            ax.plot(ri2 / au, 0.8 * np.ones_like(ri2), '+', c='C2')
            ax.set_xlim(xlim)
            ax.set_ylim(0.5, 1.5)

        return {'r': r2, 'ri': ri2, 'rho': rho2, 'tau': tau2, 'temp': temp2}
    else:
        warnings.warn('no refinement was done, please check if your disk is reasonable (radially optically thin?)')
        return {'r': r, 'ri': ri, 'rho': rho, 'tau': tau1, 'temp': temp}


def read_data(ddens=False, dtemp=False, dstat=False, dheat=False, gas=False, meanint=False, mirror=False, ilayer=None, ddir='.'):
    """
    Read RADMC3D data:

    Keywords:
    ---------

    ddens : bool
        whether or not to read in the dust density

    dtemp : bool
        whether or not to read in the dust temperature

    dstat : bool
        whether or not to read in the photon statistics

    dheat : bool
        whether or not to read in the heat sources

    gas : bool
        whether or not to read in the gas density

    meanint : bool
        whether or not to read in the mean intensity

    mirror : bool
        whether or not to mirror the data at the mid-plane

    ilayer : int
        number of AMR layers

    ddir : string
        from which directory to read the data

    Output:
    -------

    returns a `data_frame` object containing the RADMC3D data
    """
    from numpy import array
    import os
    rho = None
    rhog = None
    temp = None
    heat = None
    stat = None
    jnu = None
    nspec = None
    nlam = None

    ddir = os.path.expanduser(ddir + os.sep)

    g = read_amr_grid(directory=ddir, mirror=mirror)

    if ddens:
        fid = open(ddir + 'dust_density.inp', 'r')
        iformat = int(fid.readline().strip())
        ncell = int(fid.readline().strip())
        nspec = int(fid.readline().strip())
        if ncell != g.ncellinp:
            raise NameError('Grid Problem!')
        rho = read_data_helper(g, fid, nspec, ilayer=ilayer)
        fid.close()
    if dheat:
        fid = open(ddir + 'heatsource.inp', 'r')
        iformat = int(fid.readline().strip())
        ncell = int(fid.readline().strip())
        nspec = int(fid.readline().strip())
        if ncell != g.ncellinp:
            raise NameError('Grid Problem!')
        heat = read_data_helper(g, fid, nspec, ilayer=ilayer)
        fid.close()
    if dstat:
        fid = open(ddir + 'photon_statistics.out', 'r')
        iformat = int(fid.readline().strip())
        ncell = int(fid.readline().strip())
        if ncell != g.ncellinp:
            raise NameError('Grid Problem!')
        stat = read_data_helper(g, fid, 1, ilayer=ilayer)
        fid.close()
    if gas:
        fid = open(ddir + 'gas_density.inp', 'r')
        iformat = int(fid.readline().strip())
        ncell = int(fid.readline().strip())
        nspec = int(fid.readline().strip())
        if ncell != g.ncellinp:
            raise NameError('Grid Problem!')
        rhog = read_data_helper(g, fid, nspec, ilayer=ilayer)
        fid.close()
    if dtemp:
        fid = open(ddir + 'dust_temperature.dat', 'r')
        iformat = int(fid.readline().strip())
        ncell = int(fid.readline().strip())
        nspec = int(fid.readline().strip())
        if ncell != g.ncellinp:
            raise NameError('Grid Problem!')
        temp = read_data_helper(g, fid, nspec, ilayer=ilayer)
        fid.close()
    if meanint:
        fid = open(ddir + 'mean_intensity.out', 'r')
        iformat = int(fid.readline().strip())
        if iformat != 2:
            raise NameError('iformat is not 2!')
        ncell = int(fid.readline().strip())
        nlam = int(fid.readline().strip())
        freq = array([float(fid.readline().strip())  # noqa
                     for i in nlam])  # noqa
        if ncell != g.ncellinp:
            raise NameError('Grid Problem!')
        jnu = read_data_helper(g, fid, nlam, ilayer=ilayer)
        fid.close()
    if ilayer is None:
        ilayer = -1

    return data_frame(grid=g, rho=rho, temp=temp, heat=heat, stat=stat, rhog=rhog, jnu=jnu, ilayer=ilayer, nspec=nspec, nlam=nlam, ddir=ddir)


def read_data_helper(grid, fid, nspec, ilayer=None, specinner=False):
    """
    Reads data depending on grid definition

    Arguments:
    ----------

    grid : instance of `radmc_grid`
    :    contains the grid specification of the data to be read in

    fid : file object
    :    the file-stream to read from. The header has be be read already,
         only the actual data remains in the stream.

    nspec : int
    :    number of species in the file

    Keywords:
    ---------

    ilayer : int
    :    number of layers for AMR

    specinner : bool
    :    whether the species index is the inner (default,True) index or the outer index (False)

    Output:
    -------

    data : array
    :    the numpy array containing the data
    """
    from numpy import arange, zeros, product, fromfile
    if grid.gridstyle == 0:
        #
        # Regular grid
        #
        if specinner:
            dshape = [nspec, grid.nx, grid.ny, grid.nz]
            data = fromfile(fid, dtype='float', sep=' ', count=product(
                dshape)).reshape(dshape, order='F')
            if grid.mirror != 0 and grid.coordsys == 100:
                data0 = data
                data = zeros([nspec, grid.nx, 2 * grid.ny, grid.nz])
                for iy in arange(grid.ny):
                    data[:, :, iy, :] = data0[:, :, iy, :]
                for iy in arange(grid.ny, 2 * grid.ny):
                    data[:, :, iy, :] = data0[:, :, 2 * grid.ny - iy - 1, :]
        else:
            dshape = [grid.nx, grid.ny, grid.nz, nspec]
            data = fromfile(fid, dtype='float', sep=' ', count=product(
                dshape)).reshape(dshape, order='F')
            if grid.mirror != 0 and grid.coordsys == 100:
                data0 = data
                data = zeros([grid.nx, 2 * grid.ny, grid.nz, nspec])
                for iy in arange(grid.ny):
                    data[:, iy, :, :] = data0[:, iy, :, :]
                for iy in arange(grid.ny, 2 * grid.ny):
                    data[:, iy, :, :] = data0[:, 2 * grid.ny - iy - 1, :, :]

    elif grid.gridstyle == 1:
        #
        # Oct-tree AMR grid
        #
        # Mirror style not explicitly treated here (i.e. if mirror: then
        # still only the cells above the equatorial plane are included)
        #
        raise NameError('Layer-style AMR grid: reading in was not tested yet')
        if specinner is not None:
            dshape = [nspec, grid.ncellinp]
            data = fromfile(fid, dtype='float', sep=' ', count=product(
                dshape)).reshape(dshape, order='F')
        else:
            dshape = zeros([grid.ncellinp, nspec])
            data = fromfile(fid, dtype='float', sep=' ', count=product(
                dshape)).reshape(dshape, order='F')
    elif grid.gridstyle == 10:
        #
        # Layer-style AMR grid
        #
        # Mirror style not explicitly treated here (i.e. if mirror: then
        # still only the cells above the equatorial plane are included)
        #
        raise NameError('Layer-style AMR grid: reading in was not tested yet')
        if specinner is not None:
            if ilayer is not None:
                for ilr in arange(ilayer + 1):
                    dshape = [nspec, grid.nnxyz[0, ilr], grid.nnxyz[1, ilr], grid.nnxyz[2, ilr]]
                    data = fromfile(fid, dtype='float', sep=' ', count=product(
                        dshape)).reshape(dshape, order='F')
            else:
                data = zeros([nspec, grid.nxmax, grid.nymax, grid.nzmax, grid.nlayers + 1])
                for ilr in arange(grid.nlayers + 1):
                    dshape = [nspec, grid.nnxyz[0, ilr], grid.nnxyz[1, ilr], grid.nnxyz[2, ilr]]
                    data0 = fromfile(fid, dtype='float', sep=' ', count=product(dshape)).reshape(dshape, order='F')
                    data[:, 0:grid.nnxyz[0, ilr] - 1, 0:grid.nnxyz[1, ilr] - 1, 0:grid.nnxyz[2, ilr] - 1, ilr] = data0
        else:
            if ilayer is not None:
                for ilr in arange(ilayer + 1):
                    dshape = [grid.nnxyz[0, ilr], grid.nnxyz[1, ilr], grid.nnxyz[2, ilr], nspec]
                    data = fromfile(fid, dtype='float', sep=' ', count=product(dshape)).reshape(dshape, order='F')
            else:
                data = zeros([grid.nxmax, grid.nymax, grid.nzmax, nspec, grid.nlayers + 1])
                for ilr in arange(grid.nlayers + 1):
                    dshape = [grid.nnxyz[0, ilr], grid.nnxyz[1, ilr], grid.nnxyz[2, ilr], nspec]
                    data0 = fromfile(fid, dtype='float', sep=' ', count=product(
                        dshape)).reshape(dshape, order='F')
                    data[0:grid.nnxyz[0, ilr] - 1, 0:grid.nnxyz[1, ilr] - 1, 0:grid.nnxyz[2, ilr] - 1, :, ilr] = data0
    return data


class data_frame_general(object):
    """
    General data frame object. Name and expected attributes can be set.
    """
    _expected = []
    _name = None

    def __init__(self, expected=[], name=None, **kwargs):
        """
        Initialize object by passing the possible attributes with as dictionary
        with name:description
        """
        self._expected = list(set(self._expected + expected))
        if name is not None:
            self._name = name

        # set all expected attirbutes to none

        for key in self._expected:
            setattr(self, key, None)

        for key, value in kwargs.items():
            if not hasattr(self, key) and len(self._expected) > 0:
                print('WARNING: unknown attribute %s passed to data frame constructor' % key)
            setattr(self, key, value)

    def __repr__(self, showall=False):
        """
        Prints out a meaningful description for simulation objects (gas, dust, ...).

        Arguments:
        ----------
        obj : object
            object to be represented

        Keywords:
        ---------
        showall : bool
            If true, then also underscored attributes are printed.

        Returns:
        --------
        s : str
            string representation of object
        """
        s = ''
        if self._name is not None:
            s += self._name + ':\n'
        for key, val in self.__dict__.items():

            # skip underscored names unless asked for

            if not showall and key.startswith('_'):
                continue

            # start with variable name

            s += '\n{:20s} = '.format(key)

            # now handle different variable types separately

            if type(val) in [int, float, bool]:
                s += "{}".format(val)

            elif type(val) is str:
                s += "\"{}\"".format(val)

            elif type(val) is np.ndarray:
                s += "ndarray, shape = {}".format(val.shape)

            elif val is None:
                s += "None"

            elif type(val) in [tuple, list, dict]:
                if len(str(val)) > 25:
                    s += "{} of length {}".format(type(val).__name__, len(val))
                else:
                    s += "{}".format(val)

            else:
                s += "instance of {}".format(type(val).__name__)
        return s


class data_frame(data_frame_general):
    """
    Object containing grid and density data
    """
    _expected = [
        'grid',
        'rho',
        'rhog',
        'temp',
        'heat',
        'stat',
        'jnu',
        'ilayer',
        'nspec',
        'nlam',
        'ddir'
    ]
    _name = 'RADMC-3D data'


class radmc_image(data_frame_general):
    """
    Object containing image data
    """
    _expected = [
        'nx',
        'ny',
        'nrfr',
        'sizepix_x',
        'sizepix_y',
        'image',
        'flux',
        'x',
        'y',
        'lamb',
        'radian',
        'stokes']

    _name = 'RADMC-3D Image'


def read_image(ext=None, filename=None):
    """
    Reads the rectangular telescope image produced by RADMC3D. The file name of
    the image is assumed to be image.out if no keyword is given. If keyword
    `ext` is given, the filename  'image_'+ext+'.out' is used. If keyword
    `filename` is given, it is used as the file name.

    Keywords:
    ---------

    ext : string
        Filname extension of the image file, see above

    filename : string
        file name of the image file

    Output:
    -------

    Returns a data frame containing the image data with the following attributes:
    nx,ny,nrfr,sizepix_x,sizepix_y,image,flux,x,y,lamb,radian,stokes

    The image is in erg/(s cm^2 Hz ster)

    """
    from numpy import fromfile, product, arange
    import glob
    #
    # Read from normal file, so make filename
    #
    if filename is None:
        if ext is None:
            filename = 'image.out'
        else:
            filename = 'image_' + str(ext) + '.out'
    fstr = glob.glob(filename)
    if len(fstr) == 0:
        print('Sorry, cannot find ' + filename)
        print('Presumably radmc3d exited without succes.')
        print('See above for possible error messages of radmc3d!')
        raise NameError('File not found')
    funit = open(filename)
    #
    # Read the image
    #
    iformat = fromfile(funit, dtype='int', count=1, sep=' ')[0]
    if iformat < 1 or iformat > 4:
        raise NameError('ERROR: File format of ' + filename + ' not recognized.')
    if iformat == 1 or iformat == 3:
        radian = False
    else:
        radian = True
    if iformat == 1 or iformat == 2:
        stokes = False
    else:
        stokes = True

    nx, ny = fromfile(funit, dtype=int, count=2, sep=' ')
    nf = fromfile(funit, dtype=int, count=1, sep=' ')[0]
    sizepix_x, sizepix_y = fromfile(funit, dtype=float, count=2, sep=' ')
    lamb = fromfile(funit, dtype=float, count=nf, sep=' ')
    if nf == 1:
        lamb = lamb[0]
    if stokes:
        image_shape = [4, nx, ny, nf]
    else:
        image_shape = [nx, ny, nf]
    image = fromfile(funit, dtype=float, count=product(image_shape), sep=' ').reshape(image_shape, order='F')
    funit.close()
    #
    # If the image contains all four Stokes vector components,
    # then it is useful to transpose the image array such that
    # the Stokes index is the third index, so that the first
    # two indices remain x and y
    #
    if stokes:
        if nf > 1:
            image = image[[1, 2, 0, 3]]
        else:
            image = image[[1, 2, 0]]
    #
    # Compute the flux in this image as seen at 1 pc
    #
    flux = 0.0
    if stokes:
        for ix in arange(nx):
            for iy in arange(ny):
                flux = flux + image[ix, iy, 0, :]
    else:
        for ix in arange(nx):
            for iy in arange(ny):
                flux = flux + image[ix, iy, :]
    flux = flux * sizepix_x * sizepix_y
    if not radian:
        flux = flux / pc**2
    #
    # Compute the x- and y- coordinates
    #
    x = ((arange(nx) + 0.5) / (nx * 1.) - 0.5) * sizepix_x * nx
    y = ((arange(ny) + 0.5) / (ny * 1.) - 0.5) * sizepix_y * ny
    #
    # Return all
    #
    return radmc_image(
        nx=nx,
        ny=ny,
        nrfr=nf,
        sizepix_x=sizepix_x,
        sizepix_y=sizepix_y,
        image=image.squeeze(),
        flux=flux,
        x=x,
        y=y,
        lamb=lamb,
        radian=radian,
        stokes=stokes)


class radmc_grid(data_frame_general):
    _name = 'RADMC-3D Grid'
    _expected = []


def read_amr_grid(directory='.', basic=False, mirror=False):
    from numpy import pi, append, arange, zeros, array, sqrt
    import glob
    import os
    count1 = len(glob.glob(directory + os.sep + 'amr_grid.inp'))
    count2 = len(glob.glob(directory + os.sep + 'amr_grid.uinp'))
    count3 = len(glob.glob(directory + os.sep + 'amr_grid.binp'))
    count = count1 + count2 + count3
    if count != 1:
        raise NameError('ERROR: Need 1 and only 1 file amr_grid.*inp...')
    nx = 0
    ny = 0
    nz = 0
    nxmax = 0
    nymax = 0
    nzmax = 0
    # levelmax        = 0
    # nrleafsmax      = 0
    nrbranchesmax = 0
    nlayers = 0
    octtree = 0
    iparent = 0
    ixyz = 0
    nxyz = 0
    nnxyz = 0
    amrstyle = 0
    coordsys = 0
    # gridinfo        = 0
    # iformat         = 0
    # iprecis         = 0
    incd = [0, 0, 0]
    layer_xi = 0.0
    layer_yi = 0.0
    layer_zi = 0.0
    layer_x = 0.0
    layer_y = 0.0
    layer_z = 0.0
    if count1 > 0:
        fid = open(directory + os.sep + 'amr_grid.inp', 'r')
        #
        # For now do things simple
        #
        iformat = int(fid.readline().strip())  # noqa
        amrstyle = int(fid.readline().strip())
        coordsys = int(fid.readline().strip())
        gridinfo = int(fid.readline().strip())  # noqa
        incd = [int(l) for l in fid.readline().split()]
        nx, ny, nz = [int(l) for l in fid.readline().split()]
        if amrstyle == 0:
            #
            # Regular grid
            #
            pass
        elif amrstyle == 1:
            #
            # Oct-tree style AMR
            #
            levelmax, nrleafsmax, nrbranchesmax = [int(l) for l in fid.readline().split()]  # noqa
        elif amrstyle == 10:
            #
            # Layer style AMR
            #
            levelmax, nlayers = [int(l) for l in fid.readline().split()]  # noqa
        nxmax = nx
        nymax = ny
        nzmax = nz
        xi = zeros(nx + 1)
        yi = zeros(ny + 1)
        zi = zeros(nz + 1)
        xi = array([float(fid.readline().strip()) for i in arange(nx + 1)])
        yi = array([float(fid.readline().strip()) for i in arange(ny + 1)])
        zi = array([float(fid.readline().strip()) for i in arange(nz + 1)])
        if basic:
            #
            # Only return the basic information of the grid
            #
            fid.close()
            if coordsys < 100:
                #
                # Cartesian coordinates
                #
                x = 0.5 * (xi[0:nx] + xi[1:nx + 1])
                y = 0.5 * (yi[0:ny] + yi[1:ny + 1])
                z = 0.5 * (zi[0:nz] + zi[1:nz + 1])
                return radmc_grid(x=x, y=y, z=z, xi=xi, yi=yi, zi=zi, nx=nx,
                                  ny=ny, nz=nz, mirror=mirror, coordsys=coordsys,
                                  incx=incd[0], incy=incd[1], incz=incd[2],
                                  nlayers=nlayers, gridstyle=amrstyle)
            else:
                #
                # Spherical coordinates
                #
                ri = xi
                thetai = yi
                phii = zi
                nr = nx
                ntheta = ny
                nphi = nz
                if mirror:
                    thetai = append(thetai, pi - thetai[0:ny][::-1])
                    ntheta = ntheta * 2  # I deliberately do not increase ny; only ntheta
                r = sqrt(ri[0:nr] * ri[1:nr + 1])
                theta = 0.5 * (thetai[0:ntheta] + thetai[1:ntheta + 1])
                phi = 0.5 * (phii[0:nphi] + phii[1:nphi + 1])
                return radmc_grid(r=r, theta=theta, phi=phi, ri=ri, thetai=thetai,
                                  phii=phii, nr=nr, ntheta=ntheta, nphi=nphi,
                                  coordsys=coordsys, nx=nx, ny=ny, nz=nz,
                                  incx=incd[0], incy=incd[1], incz=incd[2],
                                  mirror=mirror, nlayers=nlayers,
                                  gridstyle=amrstyle)
        if amrstyle == 0:
            #
            # Regular grid
            #
            ncell = nx * ny * nz
            ncellinp = ncell
        elif amrstyle == 1:
            #
            # Oct-tree style AMR
            #
            octtree = zeros(nrbranchesmax, dtype='bool')  # XXX
            octtree = array([bool(fid.readline().strip()) for i in arange(nrbranchesmax)])
            ncell = sum(octtree == 0)
            ncellinp = ncell
        elif amrstyle == 10:
            #
            # Layer style AMR
            #
            iparent = zeros(nlayers + 1, dtype='int32')
            ixyz = zeros([3, nlayers + 1], dtype='int32')
            nxyz = zeros([3, nlayers + 1], dtype='int32')
            nnxyz = zeros([3, nlayers + 1], dtype='int32')
            nnxyz[:, 0] = [nx, ny, nz]
            idat = zeros(7, dtype='int32')
            ncell = nx * ny * nz
            ncellinp = nx * ny * nz
            for i in arange(nlayers + 1):
                raise NameError(
                    'AMR Not yet Implemented, see commented-out code')
                # readf,1,idat# XXX
                iparent[i] = idat[0]
                ixyz[:, i] = idat[1:4]
                nxyz[:, i] = idat[4:7]
                nnxyz[:, i] = idat[4:7]
                if incd[0] == 1:
                    nnxyz[0, i] = nnxyz[0, i] * 2
                if incd[1] == 1:
                    nnxyz[1, i] = nnxyz[1, i] * 2
                if incd[2] == 1:
                    nnxyz[2, i] = nnxyz[2, i] * 2
                ncell = ncell + nnxyz[0, i] * nnxyz[1, i] * \
                    nnxyz[2, i] - nxyz[0, i] * nxyz[1, i] * nxyz[2, i]
                ncellinp = ncellinp + nnxyz[0, i] * nnxyz[1, i] * nnxyz[2, i]
                nxmax = max([nxmax, nnxyz[0, i]])
                nymax = max([nymax, nnxyz[1, i]])
                nzmax = max([nzmax, nnxyz[2, i]])
        fid.close()
    else:
        if count2 > 0:
            raise NameError(
                'unformatted input is not implemented, see commented-out code')
            # XXX all of this reading in is unconverted & untested, array indices ARE converted
            # openr,1,'amr_grid.uinp',/f77_unformatted #XXX
            #
            # For now do things simple
            #
            # readu,1,iformat  # XXX
            # readu,1,amrstyle # XXX
            # readu,1,coordsys # XXX
            # readu,1,gridinfo # XXX
            # readu,1,incd     # XXX
            # readu,1,nx,ny,nz # XXX
            if amrstyle == 0:
                #
                # Regular grid
                #
                pass
            elif amrstyle == 1:
                #
                # Oct-tree style AMR
                #
                # readu,1,levelmax,nrleafsmax,nrbranchesmax # XXX
                pass  # XXX remove when line above is implemented
            elif amrstyle == 10:
                #
                # Layer style AMR
                #
                # readu,1,levelmax,nlayers # XXX
                pass  # XXX remove when line above is implemented
            nxmax = nx
            nymax = ny
            nzmax = nz
            xi = zeros(nx + 1)
            yi = zeros(ny + 1)
            zi = zeros(nz + 1)
            # readu,1,xi # XXX
            # readu,1,yi # XXX
            # readu,1,zi # XXX
            if basic:
                #
                # Only return the basic information of the grid
                #
                fid.close()
                if coordsys < 100:
                    #
                    # Cartesian coordinates
                    #
                    x = 0.5 * (xi[0:nx] + xi[1:nx + 1])
                    y = 0.5 * (yi[0:ny] + yi[1:ny + 1])
                    z = 0.5 * (zi[0:nz] + zi[1:nz + 1])
                    return radmc_grid(x=x, y=y, z=z, xi=xi, yi=yi, zi=zi, nx=nx, ny=ny, nz=nz, mirror=mirror,
                                      coordsys=coordsys, incx=incd[0], incy=incd[1], incz=incd[2], nlayers=nlayers,
                                      gridstyle=amrstyle)
                else:
                    #
                    # Spherical coordinates
                    #
                    ri = xi
                    thetai = yi
                    phii = zi
                    nr = nx
                    ntheta = ny
                    nphi = nz
                    if mirror:
                        thetai = append(thetai, pi - thetai[0:ny][::-1])
                        ntheta = ntheta * 2  # I deliberately do not increase ny; only ntheta
                    r = sqrt(ri[0:nr] * ri[1:nr + 1])
                    theta = 0.5 * (thetai[0:ntheta] + thetai[1:ntheta + 1])
                    phi = 0.5 * (phii[0:nphi] + phii[1:nphi + 1])
                    return radmc_grid(r=r, theta=theta, phi=phi, ri=ri, thetai=thetai, phii=phii,
                                      nr=nr, ntheta=ntheta, nphi=nphi, coordsys=coordsys,
                                      nx=nx, ny=ny, nz=nz, incx=incd[0], incy=incd[1], incz=incd[2],
                                      mirror=mirror, nlayers=nlayers, gridstyle=amrstyle)
            if amrstyle == 0:
                #
                # Regular grid
                #
                ncell = nx * ny * nz
                ncellinp = ncell
            elif amrstyle == 1:
                #
                # Oct-tree style AMR
                #
                raise NameError(
                    'STOPPING: The reading of the oct-tree AMR in unformatted style is not yet ready...')
                octtree = zeros(nrbranchesmax, dtype='bool')
                # readu,1,octtree #XXX
                ncell = sum(octtree == 0)
                ncellinp = ncell
            elif amrstyle == 10:
                #
                # Layer style AMR
                #
                iparent = zeros(nlayers + 1, dtype='int32')
                ixyz = zeros([3, nlayers + 1], dtype='int32')
                nxyz = zeros([3, nlayers + 1], dtype='int32')
                nnxyz = zeros([3, nlayers + 1], dtype='int32')
                nnxyz[:, 0] = [nx, ny, nz]
                idat = zeros(7, dtype='int32')
                ncell = nx * ny * nz
                ncellinp = nx * ny * nz
                for i in arange(1, nlayers + 1):
                    # readu,1,idat # XXX
                    iparent[i] = idat[0]
                    ixyz[:, i] = idat[1:4]
                    nxyz[:, i] = idat[4:7]
                    nnxyz[:, i] = idat[4:7]
                    if incd[0] == 1:
                        nnxyz[0, i] = nnxyz[0, i] * 2
                    if incd[1] == 1:
                        nnxyz[1, i] = nnxyz[1, i] * 2
                    if incd[2] == 1:
                        nnxyz[2, i] = nnxyz[2, i] * 2
                    ncell = ncell + nnxyz[0, i] * nnxyz[1, i] * \
                        nnxyz[2, i] - nxyz[0, i] * nxyz[1, i] * nxyz[2, i]
                    ncellinp = ncellinp + \
                        nnxyz[0, i] * nnxyz[1, i] * nnxyz[2, i]
                    nxmax = max([nxmax, nnxyz[0, i]])
                    nymax = max([nymax, nnxyz[1, i]])
                    nzmax = max([nzmax, nnxyz[2, i]])
            fid.close()
        else:
            raise NameError(
                'AMR and unformatted input not implemented yet, see commented-out code')
            # openr,1,'amr_grid.binp' # XXX
            # For now do things simple
            # readu,1,iformat # XXX
            # readu,1,amrstyle # XXX
            # readu,1,coordsys # XXX
            # readu,1,gridinfo # XXX
            # readu,1,incd # XXX
            # readu,1,nx,ny,nz # XXX
            if amrstyle == 0:
                #
                # Regular grid
                #
                pass
            elif amrstyle == 1:
                #
                # Oct-tree style AMR
                #
                # readu,1,levelmax,nrleafsmax,nrbranchesmax # XXX
                pass  # XXX remove this line if the line above is implemented
            elif amrstyle == 10:
                #
                # Layer style AMR
                #
                # readu,1,levelmax,nlayers # XXX
                pass  # XXX remove this line if the line above is implemented
            nxmax = nx
            nymax = ny
            nzmax = nz
            xi = zeros(nx + 1)
            yi = zeros(ny + 1)
            zi = zeros(nz + 1)
            # readu,1,xi # XXX
            # readu,1,yi # XXX
            # readu,1,zi # XXX
            if basic:
                #
                # Only return the basic information of the grid
                #
                fid.close()
                if coordsys < 100:
                    #
                    # Cartesian coordinates
                    #
                    x = 0.5 * (xi[0:nx] + xi[1:nx + 1])
                    y = 0.5 * (yi[0:ny] + yi[1:ny + 1])
                    z = 0.5 * (zi[0:nz] + zi[1:nz + 1])
                    return radmc_grid(x=x, y=y, z=z, xi=xi, yi=yi, zi=zi, nx=nx, ny=ny, nz=nz, mirror=mirror,
                                      coordsys=coordsys, incx=incd[0], incy=incd[1], incz=incd[2], nlayers=nlayers,
                                      gridstyle=amrstyle)
                else:
                    #
                    # Spherical coordinates
                    #
                    ri = xi
                    thetai = yi
                    phii = zi
                    nr = nx
                    ntheta = ny
                    nphi = nz
                    if mirror:
                        thetai = append(thetai, pi - thetai[0:ny][::-1])
                        ntheta = ntheta * 2  # I deliberately do not increase ny; only ntheta
                    r = sqrt(ri[0:nr] * ri[1:nr + 1])
                    theta = 0.5 * (thetai[0:ntheta] + thetai[1:ntheta + 1])
                    phi = 0.5 * (phii[0:nphi] + phii[1:nphi + 1])
                    return radmc_grid(r=r, theta=theta, phi=phi, ri=ri, thetai=thetai, phii=phii,
                                      nr=nr, ntheta=ntheta, nphi=nphi, coordsys=coordsys,
                                      nx=nx, ny=ny, nz=nz, incx=incd[0], incy=incd[1], incz=incd[2],
                                      mirror=mirror, nlayers=nlayers, gridstyle=amrstyle)
            if amrstyle == 0:
                #
                # Regular grid
                #
                ncell = nx * ny * nz
                ncellinp = ncell
            elif amrstyle == 1:
                #
                # Oct-tree style AMR
                #
                raise NameError(
                    'STOPPING: The reading of the oct-tree AMR in unformatted style is not yet ready...')
                octtree = zeros(nrbranchesmax, dtype='bool')
                # readu,1,octtree # XXX
                ncell = sum(octtree == 0)
                ncellinp = ncell
            elif amrstyle == 10:
                #
                # Layer style AMR
                #
                iparent = zeros(nlayers + 1, dtype='int32')
                ixyz = zeros([3, nlayers + 1], dtype='int32')
                nxyz = zeros([3, nlayers + 1], dtype='int32')
                nnxyz = zeros([3, nlayers + 1], dtype='int32')
                nnxyz[:, 0] = [nx, ny, nz]
                idat = zeros(7, dtype='int32')
                ncell = nx * ny * nz
                ncellinp = nx * ny * nz
                for i in arange(1, nlayers + 1):
                    raise NameError(
                        'AMR and unformatted input not implemented yet. See commented-out code')
                    # readu,1,idat # XXX
                    iparent[i] = idat[0]
                    ixyz[:, i] = idat[1:4]
                    nxyz[:, i] = idat[4:7]
                    nnxyz[:, i] = idat[4:7]
                    if incd[0] == 1:
                        nnxyz[0, i] = nnxyz[0, i] * 2
                    if incd[1] == 1:
                        nnxyz[1, i] = nnxyz[1, i] * 2
                    if incd[2] == 1:
                        nnxyz[2, i] = nnxyz[2, i] * 2
                    ncell = ncell + nnxyz[0, i] * nnxyz[1, i] * \
                        nnxyz[2, i] - nxyz[0, i] * nxyz[1, i] * nxyz[2, i]
                    ncellinp = ncellinp + \
                        nnxyz[0, i] * nnxyz[1, i] * nnxyz[2, i]
                    nxmax = max([nxmax, nnxyz[0, i]])
                    nymax = max([nymax, nnxyz[1, i]])
                    nzmax = max([nzmax, nnxyz[2, i]])
            fid.close()
    #
    # Some post-processing
    #
    if amrstyle == 10:
        layer_xi = zeros([nxmax + 1, nlayers + 1])
        layer_yi = zeros([nymax + 1, nlayers + 1])
        layer_zi = zeros([nzmax + 1, nlayers + 1])
        layer_x = zeros([nxmax, nlayers + 1])
        layer_y = zeros([nymax, nlayers + 1])
        layer_z = zeros([nzmax, nlayers + 1])
        layer_xi[0:nx + 1, 0] = xi
        layer_yi[0:ny + 1, 0] = yi
        layer_zi[0:nz + 1, 0] = zi
        layer_x[0:nx, 0] = 0.5 * (xi[0:nx] + xi[1:nx + 1])
        layer_y[0:ny, 0] = 0.5 * (yi[0:ny] + yi[1:ny + 1])
        layer_z[0:nz, 0] = 0.5 * (zi[0:nz] + zi[1:nz + 1])
        for i in arange(1, nlayers + 1):
            for k in arange(0, nnxyz[0, i] + 1, 2):
                layer_xi[k, i] = layer_xi[ixyz[0, i] - 1 + k / 2, iparent[i]]
            for k in arange(0, nnxyz[1, i] + 1, 2):
                layer_yi[k, i] = layer_yi[ixyz[1, i] - 1 + k / 2, iparent[i]]
            for k in arange(0, nnxyz[2, i] + 1, 2):
                layer_zi[k, i] = layer_zi[ixyz[2, i] - 1 + k / 2, iparent[i]]
            if coordsys < 100:
                for k in arange(1, nnxyz[0, i], 2):
                    layer_xi[k, i] = 0.5 * (layer_xi[ixyz[0, i] - 1 + (k - 1) / 2, iparent[i]] + layer_xi[ixyz[0, i] - 1 + (k + 1) / 2, iparent[i]])
            else:
                for k in arange(1, nnxyz[0, i], 2):
                    layer_xi[k, i] = sqrt(layer_xi[ixyz[0, i] - 1 + (k - 1) / 2, iparent[i]] * layer_xi[ixyz[0, i] - 1 + (k + 1) / 2, iparent[i]])
            for k in arange(1, nnxyz[1, i], 2):
                layer_yi[k, i] = 0.5 * (layer_yi[ixyz[1, i] - 1 + (k - 1) / 2, iparent[i]] + layer_yi[ixyz[1, i] - 1 + (k + 1) / 2, iparent[i]])
            for k in arange(1, nnxyz[2, i], 2):
                layer_zi[k, i] = 0.5 * (layer_zi[ixyz[2, i] - 1 + (k - 1) / 2, iparent[i]] + layer_zi[ixyz[2, i] - 1 + (k + 1) / 2, iparent[i]])
            if coordsys < 100:
                for k in arange(0, nnxyz[0, i]):
                    layer_x[k, i] = 0.5 * (layer_xi[k, i] + layer_xi[k + 1, i])
            else:
                for k in arange(0, nnxyz[0, i]):
                    layer_x[k, i] = sqrt(layer_xi[k, i] * layer_xi[k + 1, i])
            for k in arange(0, nnxyz[1, i]):
                layer_y[k, i] = 0.5 * (layer_yi[k, i] + layer_yi[k + 1, i])
            for k in arange(0, nnxyz[2, i]):
                layer_z[k, i] = 0.5 * (layer_zi[k, i] + layer_zi[k + 1, i])
    if coordsys < 100:
        #
        # Cartesian coordinates
        #
        x = 0.5 * (xi[0:nx] + xi[1:nx + 1])
        y = 0.5 * (yi[0:ny] + yi[1:ny + 1])
        z = 0.5 * (zi[0:nz] + zi[1:nz + 1])
        return radmc_grid(x=x, y=y, z=z, xi=xi, yi=yi, zi=zi, nx=nx, ny=ny, nz=nz,
                          nxmax=nxmax, nymax=nymax, nzmax=nzmax, mirror=mirror,
                          ncell=ncell, ncellinp=ncellinp, coordsys=coordsys, octtree=octtree,
                          incx=incd[0], incy=incd[1], incz=incd[2], nlayers=nlayers,
                          ixyz=ixyz, nxyz=nxyz, nnxyz=nnxyz, iparent=iparent, gridstyle=amrstyle,
                          layer_xi=layer_xi, layer_yi=layer_yi, layer_zi=layer_zi,
                          layer_x=layer_x, layer_y=layer_y, layer_z=layer_z)
    else:
        #
        # Spherical coordinates
        #
        ri = xi
        thetai = yi
        phii = zi
        nr = nx
        ntheta = ny
        nphi = nz
        if mirror:
            thetai = append(thetai, pi - thetai[0:ny][::-1])
            ntheta = ntheta * 2  # I deliberately do not increase ny; only ntheta
        r = sqrt(ri[0:nr] * ri[1:nr + 1])
        theta = 0.5 * (thetai[0:ntheta] + thetai[1:ntheta + 1])
        phi = 0.5 * (phii[0:nphi] + phii[1:nphi + 1])
        return radmc_grid(r=r, theta=theta, phi=phi, ri=ri, thetai=thetai, phii=phii,
                          nr=nr, ntheta=ntheta, nphi=nphi, ncell=ncell, coordsys=coordsys,
                          nx=nx, ny=ny, nz=nz, incx=incd[0], incy=incd[1], incz=incd[2],
                          octtree=octtree, ncellinp=ncellinp,
                          nxmax=nxmax, nymax=nymax, nzmax=nzmax, mirror=mirror, nlayers=nlayers,
                          ixyz=ixyz, nxyz=nxyz, nnxyz=nnxyz, iparent=iparent, gridstyle=amrstyle,
                          layer_xi=layer_xi, layer_yi=layer_yi, layer_zi=layer_zi,
                          layer_x=layer_x, layer_y=layer_y, layer_z=layer_z)
