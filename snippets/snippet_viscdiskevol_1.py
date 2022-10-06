from snippet_header import DiskRadialModel, np, plt, MS, year, au, finalize

d = DiskRadialModel(rout=1000 * au, alpha=1e-3)
d.make_disk_from_m_pl(mdisk=0.01 * MS, rdisk=1 * au)

plt.figure()
plt.plot(d.r / au, d.sigma)
plt.xscale('log')
plt.yscale('log')

ntime = 10
tstart = 1e3 * year     # Initial time at 1000 year (for log-spaced timesteps)
tend = 1e7 * year       # Final time at 10 Myr
time = tstart * (tend / tstart)**(np.linspace(0., 1.,
                                              ntime + 1))   # Logarithmic spacing
for itime in range(1, ntime + 1):
    d.sigma = d.get_viscous_evolution_next_timestep(
        time[itime] - time[itime - 1])
    plt.plot(d.r / au, d.sigma)
plt.ylim(1e-8, 1e6)
plt.xlabel('r [au]')
plt.ylabel(r'$\Sigma_g$')

finalize(results=(d.sigma))
