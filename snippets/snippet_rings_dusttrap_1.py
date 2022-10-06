from snippet_header import DiskRadialModel, np, plt, year, au, finalize
import copy

tstart    = 1e1 * year
tend      = 3e6 * year
ntime     = 100
nr        = 2000
alpha     = 1e-2
agrain    = 1e-2
zeta      = 2.0
ampl      = 1.0
time      = tstart * (tend / tstart)**(np.linspace(0., 1., ntime + 1))

# setup

d = DiskRadialModel(rin=10 * au, rout=1000 * au, nr=nr, alpha=alpha)
d.make_disk_from_simplified_lbp(1e0, 100 * au, 1)
d.add_dust(agrain=agrain)
lamwav = zeta * d.hp
integrand = 2 * np.pi / lamwav
integrand = 0.5 * (integrand[1:] + integrand[:-1])
phase = np.zeros(nr)
for ir in range(1, nr):
    phase[ir] = phase[ir - 1] + integrand[ir - 1] * (d.r[ir] - d.r[ir - 1])
wave = ampl * np.sin(phase)
d.alpha *= np.exp(wave)
dlist = [copy.deepcopy(d)]

# iteration

for itime in range(1, ntime + 1):
    dt = time[itime] - time[itime - 1]
    d.compute_viscous_evolution_and_dust_drift_next_timestep(dt)
    dlist.append(copy.deepcopy(d))

# plotting

plt.figure()
for itime in range(0, ntime, 20):
    plt.plot(dlist[itime].r / au,
             dlist[itime].dust[0].sigma / dlist[itime].sigma)
plt.yscale('log')
plt.xlim(20, 100)
plt.ylim(1e-6, 1e0)
plt.xlabel('r [au]')
plt.ylabel(r'$\Sigma_d/\Sigma_g$')

plt.figure()
for itime in range(0, ntime, 20):
    plt.plot(dlist[itime].r / au, dlist[itime].dust[0].sigma)
plt.yscale('log')
plt.xlim(70, 82)
plt.ylim(1e-8, 1e-1)
plt.xlabel('r [au]')
plt.ylabel(r'$\Sigma_d$')

finalize(results=(dlist[-1].dust[0].sigma))
