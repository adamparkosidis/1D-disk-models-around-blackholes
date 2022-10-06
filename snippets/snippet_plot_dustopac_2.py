from disklab.grainmodel import GrainModel
from snippet_header import np, plt, finalize

grain = GrainModel()
grain.read_opacity('dustkappa_silicate.inp')
temp    = 1.0 * 2e3**np.linspace(0., 1., 100)
kapross = grain.rosselandmean(temp)
kappl   = grain.planckmean(temp)

plt.figure()
plt.plot(temp, kappl, label='Planck')
plt.plot(temp, kapross, label='Rosseland')
plt.xlabel(r'$T [\mathrm{K}]$')
plt.ylabel(r'$\kappa_{\mathrm{abs}} [\mathrm{cm}^2/\mathrm{g}]$')
plt.xscale('log')
plt.yscale('log')
plt.xlim(temp.min(), temp.max())
plt.legend(loc='upper left')

finalize()
