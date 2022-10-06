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
    if fixedpar is not None:
        if 'mstar' in fixedpar: mstar=fixedpar['mstar']
        if 'lstar' in fixedpar: lstar=fixedpar['lstar']
    #
    # Variable parameters (sliders of widget)
    #
    tend      = param[0]
    mdisk0    = param[1]
    rdisk0    = param[2]
    alpha     = param[3]
    agrain    = param[4]
    #
    # Model setup
    #
    disk      = DiskRadialModel(mstar=mstar,lstar=lstar,rgrid=rau*au)
    disk.make_disk_from_lbp_alpha(mdisk0,rdisk0,alpha,1*year)
    disk.add_dust(agrain=agrain)
    #
    # Run the model
    #
    ntime     = 100
    time      = np.linspace(0.,tend,ntime+1)   # Linear time intervals
    for itime in range(1,ntime+1):
        dt = time[itime]-time[itime-1]
        disk.compute_viscous_evolution_and_dust_drift_next_timestep(dt)
    #
    # Return the result (you can return whatever you want to plot; here
    # we plot two results simultaneously: the gas and dust surface
    # densities)
    #
    return np.array([disk.sigma, disk.dust[0].sigma])

#
# Create the plot we wish to make interactive
#
xmin     = 1.0
xmax     = 1e3
rau      = xmin * (xmax/xmin)**np.linspace(0.,1.,100)
par      = [1e6*year,1e-2*MS,3*au,1e-3,1e-4]
fixedpar = {'mstar':MS,'lstar':LS}
result   = modelfunc(rau,par,fixedpar=fixedpar)
sigmagas = result[0]
sigmadust= result[1]
ymin     = 1e-5
ymax     = 1e+3
fig      = plt.figure()
ax       = plt.axes(xlim=(xmin,xmax),ylim=(ymin,ymax))
axgas,   = ax.loglog(rau,sigmagas,linewidth=2,label='Gas')
axdust,  = ax.loglog(rau,sigmadust,'--',linewidth=2,label='Dust')
axmodel  = [axgas,axdust]
plt.xlabel(r'$r [\mathrm{au}]$')
plt.ylabel(r'$\Sigma [\mathrm{g}/\mathrm{cm}^2]$')
plt.legend()
#
# Now make the plot interactive with sliders. We have to
# specify the slider names, the possible values and (optionally)
# the units. Then call interactive_plot() to bring it to life,
# with the function modelfunction() (see above) being the life-giver.
# Type interactive_plot? to find out more about interactive_plot():
# there are numerous examples in the document string.
#
parnames = ['tend =','mdisk0 =','rdisk0 =','alpha =','agrain = ']
params   = [1e1*1e6**np.linspace(0.,1.,100)*year,
            1e-4*1e3**np.linspace(0.,1.,100)*MS,
            np.linspace(1.,30.,100)*au,
            1e-6*1e6**np.linspace(0.,1.,100),
            1e-5*1e6**np.linspace(0.,1.,100)]
parunits = [year,MS,au,1.,1.]
ipar     = interactive_plot(rau,modelfunc,params,parnames=parnames,
                            parunits=parunits,fixedpar=fixedpar,
                            parstart=par,plotbutton=True,
                            fig=fig,ax=ax,axmodel=axmodel,returnipar=True)

finalize([])
