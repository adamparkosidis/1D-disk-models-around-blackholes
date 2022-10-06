from disklab.grainmodel import GrainModel
from snippet_header import plt, finalize

grain = GrainModel()

plt.figure()
grain.load_standard_opacity('ddn01', 'silicate')
plt.plot(grain.opac_lammic, grain.opac_kabs, label='silicate')
grain.load_standard_opacity('ddn01', 'carbon')
plt.plot(grain.opac_lammic, grain.opac_kabs, label='carbon')
grain.load_standard_opacity('ddn01', 'waterice')
plt.plot(grain.opac_lammic, grain.opac_kabs, label='waterice')
grain.load_standard_opacity('ddn01', 'crystforsterite')

plt.plot(grain.opac_lammic, grain.opac_kabs, label='crystforsterite')
plt.title('DDN01 opacities for grain radius {0:4.2f} micron'.format(grain.agrain * 1e4))
plt.xlabel(r'$\lambda [\mu\mathrm{m}]$')
plt.ylabel(r'$\kappa_{\mathrm{abs}} [\mathrm{cm}^2/\mathrm{g}]$')
plt.xscale('log')
plt.yscale('log')
plt.xlim(1e-1, 1e4)
plt.ylim(bottom=1e-3)
plt.legend(loc='upper right')

finalize()
