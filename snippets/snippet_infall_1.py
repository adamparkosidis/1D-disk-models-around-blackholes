from snippet_header import DiskRadialModel, np, plt, MS, year, au, kk, mp, finalize
import copy
rin     = 0.1 * au
alpha   = 1e-2
mdisk0  = 1e-20 * MS
rdisk0  = 1. * au
mcloud  = 0.515 * MS
mstar0  = 1e-4*mcloud
tcloud  = 14.
omcloud = 2.3e-14
tstart  = 1e0 * year
tend    = 1e7 * year
ntime   = 1000
nr      = 1200

cscloud = np.sqrt(kk * tcloud / (2.3 * mp))
time = tstart * (tend / tstart)**(np.linspace(0., 1., ntime + 1))
d = DiskRadialModel(mstar=mstar0, rin=rin, rout=1e6 * au, alpha=alpha, nr=nr)
d.make_disk_from_lbp_alpha(mdisk0, rdisk0, alpha, tstart)
dlist = [copy.deepcopy(d)]
mdisk = np.zeros(ntime + 1)
mstar = np.zeros(ntime + 1)
rcentr = np.zeros(ntime + 1)
mdot = np.zeros(ntime + 1)
mdotdsk = np.zeros(ntime + 1)
mdotcap = np.zeros(ntime + 1)
mdisk[0] = mdisk0
mstar[0] = mstar0
rcentr[0] = 0.
mdot[0] = 0.
mdotdsk[0] = 0.
mdotcap[0] = 0.
for itime in range(1, ntime + 1):
    dt = time[itime] - time[itime - 1]
    d.compute_shuulrich_infall(mcloud, cscloud, omcloud, time[itime], idistr=1)
    d.sigdot = d.infall_sigdot
    d.compute_viscous_evolution_next_timestep(dt)
    d.compute_mdot_at_interfaces()
    mdt = d.infall_mdotcap + d.mdot[0]  # Accretion rate onto star
    mst = d.mstar + mdt * dt
    d.update_stellar_mass(mst)
    d.compute_mass()
    dlist.append(copy.deepcopy(d))
    mstar[itime] = d.mstar
    mdisk[itime] = d.mass
    rcentr[itime] = d.infall_rcentr
    mdot[itime] = mdt
    mdotdsk[itime] = d.mdot[0]
    mdotcap[itime] = d.infall_mdotcap

print("Mass conservation: (Mstar+Mdisk)/(Mcloud+Mstar0+Mdisk0) = {}".format(
    (d.mstar + d.mass) / (mcloud + mstar0 + mdisk0)))

plt.figure()
for itime in range(0, ntime, 10):
    plt.plot(dlist[itime].r / au, dlist[itime].sigma)
plt.xlabel(r'$r [\mathrm{au}]$')
plt.ylabel(r'$\Sigma_g [\mathrm{g/cm^2}]$')
plt.xscale('log')
plt.yscale('log')
plt.xlim(right=1e4)
plt.ylim(bottom=1e-5)


plt.figure()
for itime in range(10, ntime, 10):
    plt.plot(dlist[itime].r / au, dlist[itime].infall_sigdot + 1e-50)
plt.xlabel(r'$r [\mathrm{au}]$')
plt.ylabel(r'$\dot\Sigma_g [\mathrm{g/cm^2/s}]$')
plt.xscale('log')
plt.yscale('log')
plt.xlim(right=1e4)
plt.ylim(bottom=1e-10)

plt.figure()
plt.plot(time / year / 1e6, mstar / MS, label='Star')
plt.plot(time / year / 1e6, mdisk / MS, label='Disk')
plt.xlabel(r'$\mathrm{time} [\mathrm{Myr}]$')
plt.ylabel(r'$M [\mathrm{M}_{\odot}]$')
plt.xscale('log')
plt.xlim(left=0.02, right=10.)
plt.legend(loc='upper left')

finalize(results=(dlist[-1].sigma,mstar,mdisk))
