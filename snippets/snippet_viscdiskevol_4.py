from snippet_header import DiskRadialModel, np, plt, MS, year, au, finalize

d=DiskRadialModel(rout=1000*au,alpha=1e-3,flang=0.02)
d.make_disk_from_m_pl(mdisk=0.01*MS,rdisk=1*au)
d.add_dust(agrain=1e-4)
kappa=1e3
d.meanopacitymodel=['supersimple',{'dusttogas':0.01,'kappadust':kappa}]
plt.figure(1)
plt.plot(d.r/au,d.sigma)
plt.xscale('log')
plt.yscale('log')
plt.ylim((1e-2,1e6))
plt.xlabel('r [au]')
plt.ylabel(r'$\Sigma_g$')
plt.figure(2)
plt.plot(d.r/au,d.tmid)
plt.xscale('log')
plt.yscale('log')
plt.ylim((1,2e3))
plt.xlabel('r [au]')
plt.ylabel(r'$T_{\mathrm{mid}}\,[\mathrm{K}]$')
ntime=100
tstart=1e3*year     # Initial time at 1000 year (for log-spaced timesteps)
tend=1e6*year       # Final time at 10 Myr
time=tstart * (tend/tstart)**(np.linspace(0.,1.,ntime+1))   # Logarithmic spacing
for itime in range(1,ntime+1):
   d.compute_mean_opacity()
   d.compute_disktmid(vischeat=True)
   d.tmid[d.tmid>1.5e3]=1.5e3
   d.compute_cs_and_hp()
   d.sigma=d.get_viscous_evolution_next_timestep(time[itime]-time[itime-1],sigma_innerbc=1e-4)
   plt.figure(1)
   plt.plot(d.r/au,d.sigma)
   plt.figure(2)
   plt.plot(d.r/au,d.tmid)

finalize(results=(d.sigma,d.tmid))
