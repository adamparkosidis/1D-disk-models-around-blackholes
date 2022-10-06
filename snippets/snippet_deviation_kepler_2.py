from snippet_header import DiskRadialModel, np, plt, MS, year, au, finalize

d = DiskRadialModel(mdot=1e-8 * MS / year, rin=10 * au, rout=30 * au, nr=1000)

rbump    = 20. * au                     # Radial location of bump
abump    = 1.0                          # Amplitude of bump (relative)
hpbump   = np.interp(rbump, d.r, d.hp)  # Pressure scale height
wbump    = 0.5 * hpbump                 # Width (stand dev) of Gaussian bump
fact     = 1 + abump * np.exp(-0.5 * ((d.r - rbump) / wbump)**2)
d.sigma *= fact
d.rhomid *= fact

# Compute SH with Sigma and P=Sigma*c_s^2

d.compute_solberg_hoiland(vertint=True)
SHsqdimless = d.SHsq / d.omk**2      # SH freq in units of Kepler freq

# plotting

plt.plot(d.r / au, SHsqdimless)
plt.xlabel(r'$r$ [au]')
plt.ylabel('SH frequency [Kepler]')

finalize(results=(SHsqdimless))
