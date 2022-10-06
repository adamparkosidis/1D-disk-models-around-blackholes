from snippet_header import DiskRadialModel, np, plt, year, au, Mea, finalize

agrain = 1e-1       # Middle-large grains
alpha  = 1e-2       # Viscous alpha
tstart = 1e2 * year
tend   = 1e6 * year
ntime  = 100
nr     = 1000
rdisk0 = 30 * au
sig0   = 1e0
time   = tstart * (tend / tstart)**(np.linspace(0., 1., ntime + 1))

# setup

d = DiskRadialModel(rin=1 * au, rout=1000 * au, nr=nr)
d.make_disk_from_simplified_lbp(sig0, rdisk0, 1)
rbump  = 20. * au                     # Radial location of bump
abump  = 1.0                          # Amplitude of bump (relative)
hpbump = np.interp(rbump, d.r, d.hp)  # Pressure scale height
wbump  = 1.0 * hpbump                 # Width (stand dev) of Gaussian bump
wave   = -abump * np.exp(-0.5 * ((d.r - rbump) / wbump)**2)
d.alpha *= np.exp(wave)
d.add_dust(agrain=agrain)
sigmadust_array = np.zeros((ntime + 1, len(d.r)))
sigmadust_array[0, :] = d.dust[0].sigma.copy()
mgas = np.zeros(ntime + 1)
mdust = np.zeros(ntime + 1)
d.compute_mass()
d.dust[0].compute_mass()
mgas[0] = d.mass
mdust[0] = d.dust[0].mass

# iteration

for itime in range(1, ntime + 1):
    dt = time[itime] - time[itime - 1]
    d.compute_viscous_evolution_and_dust_drift_next_timestep(dt)
    sigmadust_array[itime, :] = d.dust[0].sigma.copy()
    d.compute_mass()
    d.dust[0].compute_mass()
    mgas[itime] = d.mass
    mdust[itime] = d.dust[0].mass

f1 = plt.figure()
plt.plot(time / year, mgas / Mea, label='gas')
plt.plot(time / year, mdust / Mea, label='dust')
plt.xlabel('t [year]')
plt.ylabel(r'$M\; [M_{\mathrm{\oplus}}]$')
plt.xscale('log')
plt.yscale('log')
plt.legend()

f2 = plt.figure()
d.plot(d.dust[0].rhomid / d.rhomid,
       ylabel=r'$\rho_{\mathrm{dust}}/\rho_{\mathrm{gas}}$',
       ymin=1e-4, ymax=1e2)
plt.text(1e2, 30., 't = {0:9.2e} Myr'.format(time.max() / (1e6 * year)))

d.anim(time, sigmadust_array, ymin=1e-5, pause=30, ylabel=r'$\Sigma_{\mathrm{dust}}$')

finalize([f1, f2],results=(d.sigma,d.dust[0].sigma))
