from snippet_header import DiskRadialModel, np, plt, year, au, finalize, colors

tstart = 1e3 * year
tend   = 3e6 * year
ntime  = 10
nr     = 100
nspec  = 2
time   = tstart * (tend / tstart)**(np.linspace(0., 1., ntime + 1))

d = DiskRadialModel(rout=1000 * au, nr=nr)
d.make_disk_from_simplified_lbp(1e3, 10 * au, 1)
d.add_dust(agrain=1e-4, dtg=0.1e-2)
d.add_dust(agrain=1e-1, dtg=0.9e-2)

for i, dust in enumerate(d.dust):
    plt.plot(d.r / au, dust.sigma, c=colors[i], alpha=0.1)

for itime in range(1, ntime + 1):
    dt = time[itime] - time[itime - 1]
    d.compute_viscous_evolution_and_dust_drift_next_timestep(dt)
    for i, dust in enumerate(d.dust):
        plt.plot(d.r / au, dust.sigma, c=colors[i], alpha=0.1 + 0.9 * itime / ntime)

plt.xscale('log')
plt.yscale('log')
plt.ylim(1e-6, 1e3)
plt.xlabel('r [au]')
plt.ylabel(r'$\Sigma_d$')

finalize(results=(d.dust[0].sigma,d.dust[1].sigma))
