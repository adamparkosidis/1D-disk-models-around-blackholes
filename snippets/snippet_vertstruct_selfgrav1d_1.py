from snippet_header import DiskVerticalModel, np, plt, MS, LS, au, kk, mp, GG, finalize
from copy import deepcopy

mstar  = 1 * MS
lstar  = 1 * LS
r      = 1 * au
siggas = 1700. * 10
dtg    = 0.01
kapdst = 1e2
flang  = 0.05
zrmax  = 0.2
opac   = ['supersimple', {'dusttogas': 0.01, 'kappadust': 1e2}]
vert   = DiskVerticalModel(mstar, r, siggas, flang=flang, zrmax=zrmax,
                           lstar=lstar, meanopacitymodel=opac)
vert.iterate_vertical_structure(iterate_selfgrav1d=False)
vertsg = deepcopy(vert)
vertsg.iterate_vertical_structure(iterate_selfgrav1d=True)

cs      = np.sqrt(kk*vert.tgas[0]/(2.3*mp))
omk     = np.sqrt(GG*mstar/r**3)
Qtoomre = cs * omk / (np.pi * GG * siggas)
print('Qtoomre = {}'.format(Qtoomre))

plt.figure()
plt.plot(vert.z / vert.r, vert.rhogas,label='No Selfgrav')
plt.plot(vert.z / vert.r, vertsg.rhogas,label='1-D Selfgrav')
plt.xlabel('z/r')
plt.ylabel(r'$\rho [\mathrm{g}/\mathrm{cm}^3]$')
plt.legend()

finalize(results=(vert.rhogas))
