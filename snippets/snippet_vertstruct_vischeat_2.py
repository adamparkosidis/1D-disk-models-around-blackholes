from snippet_header import DiskVerticalModel, np, plt, MS, au, LS, finalize
mstar  = 1 * MS
lstar  = 1 * LS
r      = 1 * au
siggas = 1700.
dtg    = 0.01
kapdst = 1e2
flang  = 0.05
zrmax  = 0.2
nz     = 100
alpha  = np.zeros(nz) + 1e-3
opac = ['supersimple', {'dusttogas': 0.01, 'kappadust': 1e2}]
vert = DiskVerticalModel(mstar, r, siggas, flang=flang, zrmax=zrmax,
                     lstar=lstar, meanopacitymodel=opac,
                     alphavisc=alpha, nz=nz)
vert.alphavisc[vert.z < 0.03 * vert.r] = 0.0
vert.iterate_vertical_structure()

plt.figure()
plt.plot(vert.z / vert.r, vert.rhogas)
plt.xlabel('z/r')
plt.yscale('log')
plt.ylabel(r'$\rho [\mathrm{g}/\mathrm{cm}^3]$')

plt.figure()
plt.plot(vert.z / vert.r, vert.tgas)
plt.xlabel('z/r')
plt.ylabel(r'$T [K]$')
plt.ylim(bottom=0)

finalize(results=(vert.rhogas,vert.tgas))
