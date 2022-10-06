from disklab.grainmodel import GrainModel
from snippet_header import plt, finalize
grain = GrainModel()
grain.read_opacity('dustkappa_silicate.inp')

plt.figure()
plt.plot(grain.opac_lammic, grain.opac_kabs, label='Absorption')
plt.plot(grain.opac_lammic, grain.opac_ksca, label='Scattering')
plt.xlabel(r'$\lambda [\mu\mathrm{m}]$')
plt.ylabel(r'$\kappa [\mathrm{cm}^2/\mathrm{g}]$')
plt.xscale('log')
plt.yscale('log')
plt.xlim(1e-1, 1e4)
plt.ylim(bottom=1e-3)
plt.legend(loc='upper right')

finalize()
