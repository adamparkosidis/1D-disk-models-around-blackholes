from snippet_header import np, plt, DiskRadialModel, au, year, MS, LS, finalize
from disklab.interactive_plot import *
#
# Model function in a form that can be used by the interactive_plot widget
#
def modelfunc(rau,param,fixedpar=None):
    #
    # Fixed parameters
    #
    mstar     = 1*MS  # Default stellar mass: Solar mass
    lstar     = 1*LS  # Default stellar luminosity: Solar luminosity
    lam       = 0.13  # Default wavelength
    if fixedpar is not None:
        if 'mstar' in fixedpar: mstar=fixedpar['mstar']
        if 'lstar' in fixedpar: lstar=fixedpar['lstar']
        if 'lam' in fixedpar:   lam=fixedpar['lam']
    #
    # Variable parameters (sliders of widget)
    #
    tend      = param[0]
    mdisk0    = param[1]
    rdisk0    = param[2]
    alpha     = param[3]
    #
    # Model setup
    #
    disk      = DiskRadialModel(mstar=mstar,lstar=lstar,rgrid=rau*au)
    disk.make_disk_from_lbp_alpha(mdisk0,rdisk0,alpha,1*year)
    #
    # Run the model
    #
    ntime     = 100
    time      = np.linspace(0.,tend,ntime+1)   # Linear time intervals
    for itime in range(1,ntime+1):
        dt = time[itime]-time[itime-1]
        disk.compute_viscous_evolution_next_timestep(dt)
    #
    # Compute the linear brightness temperature, assuming a given
    # dust-to-gas ratio.
    #
    disk.add_dust(agrain=1e-1)
    lam = np.array([1e-1,1e-2]) # lambda = 1.0, 0.1 mm
    disk.dust[0].grain.compute_simple_opacity(lam,tabulatemean=False)
    disk.compute_onezone_intensity(lam)
    #
    # Return the intensities at the two wavelengths (in units of
    # erg/cm^2/s/Hz/ster)
    #
    return np.vstack((disk.intensity[0,:],disk.intensity[1,:]))

#
# Create the plot we wish to make interactive
#
xmin     = 1.0
xmax     = 1e3
rau      = xmin * (xmax/xmin)**np.linspace(0.,1.,100)
par      = [1e6*year,1e-2*MS,3*au,1e-3]
fixedpar = {'mstar':MS,'lstar':LS}
intensity= modelfunc(rau,par,fixedpar=fixedpar)
ymin     = 1e-18
ymax     = 1e-8
fig      = plt.figure()
ax       = plt.axes(xlim=(xmin,xmax),ylim=(ymin,ymax))
# Plot some 'observational data'
axd0     = ax.errorbar([2,10,50],[1e-12,3e-13,1e-14],
                       xerr=[0.2,1,5],yerr=[6e-13,2e-13,6e-15],
                       fmt='o',label=r'Data $\lambda=1$ mm')
axd1     = ax.errorbar([3,20,40],[1e-10,1e-12,1e-14],
                       xerr=[0.2,2,3],yerr=[8e-11,3e-13,8e-15],
                       fmt='o',label=r'Data $\lambda=0.1$ mm')
ax.set_prop_cycle(None)  # Reset the color cycle
# Now plot two model curves
ax0,     = ax.loglog(rau,np.ones_like(rau),linewidth=2,
                     label=r'Model $\lambda=1\,\mathrm{mm}$')
ax1,     = ax.loglog(rau,np.ones_like(rau),linewidth=2,
                     label=r'Model $\lambda=0.1\,\mathrm{mm}$')
axmodel  = [ax0,ax1]
plt.xlabel(r'$r [\mathrm{au}]$')
plt.ylabel(r'$I_\nu [\mathrm{CGS}]$')
plt.legend()
#
# Now make the plot interactive with sliders. We have to
# specify the slider names, the possible values and (optionally)
# the units. Then call interactive_plot() to bring it to life,
# with the function modelfunction() (see above) being the life-giver.
# Type interactive_plot? to find out more about interactive_plot().
#
parnames = ['tend =','mdisk0 =','rdisk0 =','alpha =']
params   = [1e1*1e6**np.linspace(0.,1.,100)*year,
            1e-4*1e3**np.linspace(0.,1.,100)*MS,
            np.linspace(1.,30.,100)*au,
            1e-6*1e6**np.linspace(0.,1.,30)]
parunits = [year,MS,au,1.]
ipar     = interactive_plot(rau,modelfunc,params,parnames=parnames,
                            parunits=parunits,fixedpar=fixedpar,
                            parstart=par,plotbutton=True,
                            fig=fig,ax=ax,axmodel=axmodel,returnipar=True)

finalize([])
