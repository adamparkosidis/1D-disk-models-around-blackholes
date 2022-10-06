from snippet_header import DiskRadialModel, np, plt, MS, year, au, finalize

ntime  = np.array([1, 10, 30, 100, 300])
tstart = 1e3 * year
tend   = 3e6 * year

plt.figure()
for iter in range(len(ntime)):
    d = DiskRadialModel(rout=1000 * au, alpha=1e-3)
    d.make_disk_from_m_pl(mdisk=0.01 * MS, rdisk=1 * au)
    time = tstart * (tend / tstart)**(np.linspace(0., 1., ntime[iter] + 1))
    for itime in range(1, ntime[iter] + 1):
        d.sigma = d.get_viscous_evolution_next_timestep(
            time[itime] - time[itime - 1])
    plt.plot(d.r / au, d.sigma, label='ntime = {}'.format(ntime[iter]))

plt.xscale('log')
plt.yscale('log')
plt.ylim(1e-10, 1e4)
plt.legend()
plt.xlabel('r [au]')
plt.ylabel(r'$\Sigma_g$')

finalize(results=(d.sigma))
