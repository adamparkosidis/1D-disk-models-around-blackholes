from snippet_header import np, plt, DiskRadialModel, year, au, MS, finalize
import copy

tstart    = 1e4 * year
tend      = 2e6 * year
ntime     = 1000
nr        = 2000
alpha1    = 1e-3
alphamin  = 1e-6
rdeadin   = 0.4 * au
siglive   = 2e2    # Twice Gammie's penetration depth of cosmic rays
St        = 0.01
mdisk0    = 1e-2 * MS
rdisk0    = 10 * au

# setup

d = DiskRadialModel(rin=0.1 * au, rout=1000 * au, nr=nr)
d.make_disk_from_lbp_alpha(mdisk0, rdisk0, alpha1, tstart)
d.add_dust(St=St)
d.alpha = np.zeros(nr) + d.alpha

# iteration

time = tstart * (tend / tstart)**(np.linspace(0., 1., ntime + 1))
dlist = [copy.deepcopy(d)]
for itime in range(1, ntime + 1):
    dt = time[itime] - time[itime - 1]
    d.alpha[:] = alpha1
    d.alpha[d.sigma > siglive] *= siglive / d.sigma[d.sigma > siglive]
    d.alpha[d.r < rdeadin] = alpha1
    d.alpha[d.alpha < alphamin] = alphamin
    d.compute_viscous_evolution_and_dust_drift_next_timestep(dt, updatestokes=False)
    dlist.append(copy.deepcopy(d))

# plotting

plt.figure()
for itime in range(0, ntime + 1, ntime // 5):
    s = '{0:8.2e} years'.format(time[itime] / year)
    plt.plot(dlist[itime].r / au, dlist[itime].sigma, label=s)
plt.xscale('log')
plt.yscale('log')
plt.ylim(bottom=1e-6)
plt.xlabel('r [au]')
plt.ylabel(r'$\Sigma_g$')
plt.legend()

plt.figure()
for itime in range(0, ntime + 1, ntime // 5):
    s = '{0:8.2e} years'.format(time[itime] / year)
    plt.plot(dlist[itime].r / au, dlist[itime].dust[0].sigma, label=s)
plt.xscale('log')
plt.yscale('log')
plt.ylim(bottom=1e-6)
plt.xlabel('r [au]')
plt.ylabel(r'$\Sigma_d$')
plt.legend()

finalize()
