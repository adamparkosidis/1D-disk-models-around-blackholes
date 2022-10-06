from snippet_header import DiskRadialModel, np, plt, MS, LS, year, au, finalize

d = DiskRadialModel(mstar=MS, nr=100, flang=0.05, rin=4e-2*au, lstar = 9*LS, alpha=1e-3, tbg=5)
d.make_disk_from_powerlaw(sig0 = 3000, r0 = au, plsig = -0.5)

d.meanopacitymodel = ['belllin']

ntime   = 101
tstart  = 1e3*year # Initial time at 1000 year (for log-spaced timesteps)
tend    = 1e6*year # Final time at 1 Myr
time    = tstart * (tend/tstart)**(np.linspace(0.,1.,ntime+1)) # Logarithmic spacing

for itime in range(1,ntime+1):
    d.compute_mean_opacity()
    d.compute_disktmid(vischeat=True, fixmeanopac=False)
    d.compute_cs_and_hp()
    d.sigma=d.get_viscous_evolution_next_timestep(time[itime]-time[itime-1],sigma_innerbc=1e-2)
    d.compute_rhomid_from_sigma()

plt.figure()
plt.loglog(d.r / au, d.tmid, label='Mid')
plt.loglog(d.r / au, d.tirr, label='Irradd')
plt.loglog(d.r / au, d.tvisc, label='Viscous')
plt.xlabel('r [au]')
plt.ylabel(r'$T_{\mathrm{mid}}$')
plt.legend()

finalize(results=(d.tmid))
