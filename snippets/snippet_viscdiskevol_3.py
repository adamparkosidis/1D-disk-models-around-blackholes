from snippet_header import DiskRadialModel, np, plt, MS, year, au, finalize

alpha  = 1e-3
mdisk0 = 0.01 * MS
rdisk0 = 1 * au
tstart = 1e3 * year
tend   = 1e6 * year
ntime  = 300
nr     = 1000
time = tstart * (tend / tstart)**(np.linspace(0., 1., ntime + 1))
d = DiskRadialModel(rout=1000 * au, alpha=alpha, nr=nr)
d.make_disk_from_m_pl(mdisk0, rdisk=rdisk0)
for itime in range(1, ntime + 1):
    d.sigma = d.get_viscous_evolution_next_timestep(
        time[itime] - time[itime - 1])
lbp = DiskRadialModel(rout=1000 * au, nr=nr)
lbp.make_disk_from_lbp_alpha(mdisk0, rdisk0, alpha, tend)
plt.figure()
plt.plot(d.r / au, d.sigma, label='Numerical')
plt.plot(lbp.r / au, lbp.sigma, label='Analytic (LBP)')
plt.xscale('log')
plt.yscale('log')
plt.legend(loc='lower left')
plt.ylim(1e-16, 1e6)
plt.xlabel('r [au]')
plt.ylabel(r'$\Sigma_g$')

finalize(results=(d.sigma))
