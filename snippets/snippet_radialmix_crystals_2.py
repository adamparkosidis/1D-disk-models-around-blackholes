from snippet_header import DiskRadialModel, np, plt, MS, au, year, LS, finalize, DiskRadialComponent
import copy

St     = 1e-10       # Let's take infinitely small dust
tstart = 1e1 * year
tend   = 3e6 * year
ntime  = 100
nr     = 1000
dtg    = 0.01
tcryst = 800.
nspec  = 2
time   = tstart * (tend / tstart)**(np.linspace(0., 1., ntime + 1))

d = DiskRadialModel(mstar=2.4 * MS, lstar=30 * LS, tstar=1e4, rout=1000 * au, nr=nr)
d.make_disk_from_simplified_lbp(1e3, 3 * au, 1)
d.dust = []
d.dust.append(DiskRadialComponent(d, St=St, sigma=dtg * d.sigma))
d.dust.append(DiskRadialComponent(d, St=St, sigma=1e-20 * d.sigma))
d.dust[0].sigma[d.tmid > tcryst] = 1e-20 * d.sigma[d.tmid > tcryst]  # 'Crystallize' hot part
d.dust[1].sigma[d.tmid > tcryst] = dtg * d.sigma[d.tmid > tcryst]    # 'Crystallize' hot part
d.Sc = 3.0  # Set the Schmidt number

# iteration

dlist = [copy.deepcopy(d)]
ircryst = np.where(d.tmid > tcryst)[0][-1]
for itime in range(1, ntime + 1):
    dt = time[itime] - time[itime - 1]
    d.compute_viscous_evolution_next_timestep(dt)
    extracond = [(0., 1., 0., 1, ircryst)]
    d.dust[0].compute_dust_radial_drift_next_timestep(dt, extracond=extracond)
    extracond = [(0., 1., dtg, 1, ircryst)]
    d.dust[1].compute_dust_radial_drift_next_timestep(dt, extracond=extracond)
    dlist.append(copy.deepcopy(d))

# plotting

plt.figure()
for itime in range(0, ntime + 1, 10):
    s = '{0:8.2e} years'.format(time[itime] / year)
    plt.plot(dlist[itime].r / au, dlist[itime].dust[1].sigma / dlist[itime].sigma, label=s)

plt.plot(d.r / au, dtg * (d.r[ircryst] / d.r)**(1.5 * d.Sc), linewidth=3, linestyle='--')
plt.xscale('log')
plt.yscale('log')
plt.ylim(1e-10, 1e0)
plt.xlabel('r [au]')
plt.ylabel('crystal abundance')
plt.legend()

finalize(results=(dlist[-1].dust[1].sigma))
