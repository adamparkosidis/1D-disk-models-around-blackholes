from snippet_header import DiskVerticalModel, DiskVerticalComponent, np, plt, MS, LS, \
    au, year, finalize

mstar   = 1 * MS
lstar   = 1 * LS
r       = 1 * au
siggas  = 1700.
dtg     = 0.01
kapdst  = 1e2
flang   = 0.05
zrmax   = 0.2
opac    = ['supersimple', {'dusttogas': 0.01, 'kappadust': 1e2}]
dtg     = 0.01
agrain  = 1e0
xigrain = 3.0
alpha   = 1e-3
nz      = 1000
time    = np.array([0., 1e0, 1e1, 1e2, 1e3, 1e4, 1e5]) * year
ntime   = time.size
vert    = DiskVerticalModel(mstar, r, siggas, flang=flang, zrmax=zrmax,
                        lstar=lstar, meanopacitymodel=opac, alphavisc=alpha, nz=nz)
vert.iterate_vertical_structure()
dstvert = DiskVerticalComponent(vert, dtg, agrain=agrain, xigrain=xigrain)

plt.figure()
plt.plot(vert.z / vert.r, vert.rhogas, label='gas')
for it in range(ntime - 1):
    dtime = time[it + 1] - time[it]
    dstvert.timestep_settling_mixing(dtime)
    plt.plot(vert.z / vert.r, dstvert.rho,
             label='dust t={0:8.1e} yr'.format(time[it + 1] / year))
    print('dust surface density = {0:3.5e}'.format(dstvert.diskverticalmodel.vertically_integrate(dstvert.rho)))

rhodustfinal = dstvert.rho.copy()

dstvert.compute_settling_mixing_equilibrium()
plt.plot(vert.z / vert.r, dstvert.rho, label='dust, equilibrium')

plt.xlabel('z/r')
plt.yscale('log')
plt.ylim(bottom=1e-36)
plt.legend(loc='lower left')
plt.ylabel(r'$\rho [\mathrm{g}/\mathrm{cm}^3]$')

finalize(results=(rhodustfinal,dstvert.rho))
