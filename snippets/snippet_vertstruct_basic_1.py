from snippet_header import DiskVerticalModel, np, plt, MS, LS, au, kk, mp, finalize

mstar  = 1 * MS
lstar  = 1 * LS
r      = 1 * au
siggas = 1700.
dtg    = 0.01
kapdst = 1e2
flang  = 0.05
zrmax  = 0.2
opac   = ['supersimple', {'dusttogas': 0.01, 'kappadust': 1e2}]
vert   = DiskVerticalModel(mstar, r, siggas, flang=flang, zrmax=zrmax,
                           lstar=lstar, meanopacitymodel=opac)
vert.kappadust = kapdst

# For comparison: The simple Gaussian model:
cs       = np.sqrt(kk*vert.tgas[0]/(vert.mugas*mp))
hp       = cs/vert.omk_midpl
rhogauss = (siggas/(np.sqrt(2*np.pi)*hp))*np.exp(-0.5*(vert.z/hp)**2)

plt.figure()
plt.plot(vert.z / vert.r, vert.rhogas,label='Full isothermal solution')
plt.plot(vert.z / vert.r, rhogauss,':',label='Gaussian approximation')
plt.xlabel('z/r')
plt.yscale('log')
plt.ylabel(r'$\rho [\mathrm{g}/\mathrm{cm}^3]$')
plt.legend()

plt.figure()
plt.plot(vert.z / vert.r, vert.tgas)
plt.xlabel('z/r')
plt.ylabel(r'$T [K]$')
plt.ylim(0, 170.)

finalize(results=(vert.rhogas,vert.tgas))
