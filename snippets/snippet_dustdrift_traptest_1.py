from snippet_header import DiskRadialModel, np, plt, MS, year, au, finalize
import copy

alpha  = 1e-1            # Viscous alpha
tstart = 1e2 * year      
tend   = 1e6 * year
ntime  = 1000
r0     = 20 * au
sig0   = 1e0
tmid0  = 35.
plsig  = +2e0            # To keep p constant, choose +2.
pltmp  = -1.             # To keep hp/r constant, choose -1.
nr     = 1000
rin    = r0*0.7
rout   = r0*1.3
rbump  = r0              # Radial location of bump
abump  = 10000.0         # Relative amplitude of bump
whp    = 1.0             # Width of bump in hp units
St     = 0.01            # Stokes number at the peak

# Time array

time   = tstart * (tend / tstart)**(np.linspace(0., 1., ntime + 1))

# Setup main disk

sigbg  = sig0/abump
d      = DiskRadialModel(rin=rin, rout=rout, nr=nr, alpha=alpha)
d.tmid = tmid0*(d.r/r0)**pltmp
d.compute_cs_and_hp()
d.make_disk_from_powerlaw(sigbg, r0, plsig)
d.compute_rhomid_from_sigma()
d.pbg  = d.rhomid * d.cs**2           # Just for convenience, not necessary

# Add a bump to the pressure, and reconstruct Sigma

hpbump = np.interp(rbump, d.r, d.hp)  # Pressure scale height
wbump  = whp * hpbump                 # Width (stand dev) of Gaussian bump
gauss  = abump * np.exp(-0.5 * ((d.r - rbump) / wbump)**2)
d.pmid = d.pbg.copy() * ( 1. + gauss )
d.rhomid = d.pmid/d.cs**2
d.sigma = d.rhomid * d.hp * (2*np.pi)**0.5

# Compute the analytic solution for this Gaussian pressure profile, ignoring
# the background pressure. sigdgauss is the simple analytic solution, for
# constant St, while sigdfull is the analytic solution including the change
# in St in the wings of the bump, because rho_gas drops there. It is the
# radial version of the Fromang & Nelson 2009 solution (their Eq. 19).

sigd0      = 0.01*sig0                    # Arbitrary normalization
Sc         = 1.0                          # Schmidt number
zz         = 0.5*((d.r - rbump) / wbump)**2
zz[zz>100] = 100.
sigdfull   = sigd0 * np.exp(-(Sc*St/alpha)*(np.exp(zz)-1)-zz)
psi        = np.sqrt(alpha/(Sc*St))
wdust      = wbump / np.sqrt( 1 + psi**(-2.0) )
sigdgauss  = sigd0 * np.exp(-0.5*(d.r - rbump)**2/wdust**2)

# Now add a dust component; take it to be the analytic solution of dust
# trapping in a Gauss

d.add_dust(St=St)
d.dust[0].sigma[:] = sigdfull.copy()

# Prepare the time-dependent arrays

sigmadust_array = np.zeros((ntime + 1, len(d.r)))
sigmadust_array[0, :] = d.dust[0].sigma.copy()
mdust = np.zeros(ntime + 1)
d.dust[0].compute_mass()
mdust[0] = d.dust[0].mass

# To get the 'full' solution also for the numerical model, we have to
# allow the Stokes number to be inversely proportional to the gas density.
# For constant St the solution remains a Gauss. Comment the following
# lines out to plot the case for St=constant

rhomid0         = np.interp(r0,d.r,d.rhomid)
d.dust[0].St[:] = St / ( d.rhomid / rhomid0 )

# iteration

for itime in range(1, ntime + 1):
    dt = time[itime] - time[itime - 1]
    d.dust[0].compute_dust_radial_drift_next_timestep(dt,updatestokes=False,fixgas=True)
    sigmadust_array[itime, :] = d.dust[0].sigma.copy()
    d.dust[0].compute_mass()
    mdust[itime] = d.dust[0].mass

plt.plot(d.r/au,d.dust[0].sigma,label='Numerical solution')
plt.plot(d.r/au,sigdfull,'--',label='Analytical solution')
plt.plot(d.r/au,sigdgauss,':',label='Simplified analytical')
plt.plot(d.r/au,d.pmid/d.pmid.max()*sigdgauss.max(),label='Gas pressure profile')
plt.ylim((1e-6*sigdgauss.max()),1.2*sigdgauss.max())
plt.yscale('log')
plt.xlabel('r [au]')
plt.ylabel(r'$\Sigma_d,\;p_g$')
plt.legend()

finalize(results=(d.dust[0].sigma))

