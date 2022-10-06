from snippet_header import DiskRadialModel, np, plt, year, au, finalize
import copy
agrain = 1e-2  # Middle-large grains
tstart = 1e1 * year
tend   = 1e6 * year
ntime  = 100
nr     = 100
rdisk0 = 3 * au
sig0   = 1e3
time   = tstart * (tend / tstart)**(np.linspace(0., 1., ntime + 1))

# setup

d = DiskRadialModel(rout=1000 * au, nr=nr)
d.make_disk_from_simplified_lbp(sig0, rdisk0, 1)
d.add_dust(agrain=agrain)

# iteration

dlist = [copy.deepcopy(d)]
for itime in range(1, ntime + 1):
    dt = time[itime] - time[itime - 1]
    d.sigma = d.get_viscous_evolution_next_timestep(dt)
    d.compute_rhomid_from_sigma()
    d.dust[0].compute_stokes_from_agrain()
    d.dust[0].sigma = d.dust[0].get_dust_radial_drift_next_timestep(dt)
    # d.compute_disktmid()  # Uncomment if Tmid depends on d.sigdust
    dlist.append(copy.deepcopy(d))

# plotting

plt.figure()
for itime in range(0, ntime, 10):
    plt.plot(dlist[itime].r / au, dlist[itime].sigma)
plt.xscale('log')
plt.yscale('log')
plt.ylim(1e-8, 1e3)
plt.xlabel('r [au]')
plt.ylabel(r'$\Sigma_g$')


plt.figure()
for itime in range(0, ntime, 10):
    plt.plot(dlist[itime].r / au, dlist[itime].dust[0].sigma)
plt.xscale('log')
plt.yscale('log')
plt.ylim(1e-8, 1e3)
plt.xlabel('r [au]')
plt.ylabel(r'$\Sigma_d$')

finalize(results=(dlist[-1].sigma,dlist[-1].dust[0].sigma))
