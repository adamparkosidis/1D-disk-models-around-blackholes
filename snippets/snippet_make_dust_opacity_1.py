from disklab.makedustopac import compute_opac_mie, write_radmc3d_kappa_file
from snippet_header import np, plt, finalize
agraincm  = 10 * 1e-4     # Grain size in cm
logawidth = 0.05          # Smear out the grain size by 5% in both directions
na        = 20            # Use 20 grain size samples
optconst  = "pyrmg70"     # The optical constants name
matdens   = 3.0           # The material density in gram / cm^3

# Extrapolate optical constants beyond its wavelength grid, if necessary

extrapol     = True
verbose      = False         # If True, then write out status information
lamcm        = 10.0**np.linspace(-1, 3, 200) * 1e-4
optconstfile = optconst + '.lnk'

print("Running the code. Please wait...")
opac = compute_opac_mie(optconstfile, matdens, agraincm, lamcm,
                        extrapolate=extrapol, logawidth=logawidth, na=na)
write_radmc3d_kappa_file(opac, optconst)

# plotting

plt.figure()
plt.plot(opac['lamcm'] * 1e4, opac['kabs'], label='Absorption')
plt.plot(opac['lamcm'] * 1e4, opac['kscat'], label='Scattering')
plt.xlabel(r'$\lambda [\mu\mathrm{m}]$')
plt.ylabel(r'$\kappa [\mathrm{cm}^2/\mathrm{g}]$')
plt.xscale('log')
plt.yscale('log')
plt.xlim(1e-1, 1e3)
plt.ylim(bottom=1e-3)
plt.legend(loc='lower left')

finalize()
