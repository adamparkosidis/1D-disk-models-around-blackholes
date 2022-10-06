from disklab.grainmodel import GrainModel
from snippet_header import plt, np, finalize

grain = GrainModel()
grain.xigrain = 3.0
agr = 1e-5 * 1e4**np.linspace(0., 1., 5)

plt.figure()
for agrain in agr:
    grain.agrain = agrain
    grain.compute_simple_opacity()
    plt.plot(grain.opac_lammic, grain.opac_kabs,
             label='a = {0:6.1f} micron'.format(agrain * 1e4))
plt.xlabel(r'$\lambda [\mu\mathrm{m}]$')
plt.ylabel(r'$\kappa_{\mathrm{abs}} [\mathrm{cm}^2/\mathrm{g}]$')
plt.xscale('log')
plt.yscale('log')
plt.xlim(1e-1, 1e4)
plt.ylim(bottom=1e-3)
plt.legend(loc='lower left')
plt.title('Simple opacity model of Ivezic et al. (1997)')

finalize()
