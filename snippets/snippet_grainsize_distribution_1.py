from snippet_header import DiskRadialModel, np, plt, au, finalize

na      = 20
amin    = 1e-5     # 0.1 micron
amax    = 1e-3     # 10 micron
agraini = amin * (amax / amin)**np.linspace(0., 1., na + 1)
agrain  = 0.5 * (agraini[1:] + agraini[:-1])
xi      = 3.6      # Material density of the grain in g/cm^3
mgrain  = (4. * np.pi / 3.) * xi * agrain**3
gamma   = -3.5     # Here the example of an MRN distribution
abun    = agrain**(gamma + 4.)
abun    /= abun.sum()
rdisk0   = 3 * au  # Radius of the disk
sig0     = 1e3     # Gas surface density at rdisk0

d = DiskRadialModel()
d.make_disk_from_simplified_lbp(sig0, rdisk0, 1)
for ia in range(na):
    d.add_dust(agrain=agrain[ia], xigrain=xi, dtg=abun[ia])
sigmadust = d.dust[0].join_multi_array(d.dust)  # Make a 2-D array: Sigma(r,a)
dlna = np.log(agraini[1]) - np.log(agraini[0])  # Only valid for log grid in a

# Converted Sigma into dSigma/dln(a) = distr function

dsigdlna = sigmadust / dlna

# plotting

plt.figure()
plt.plot(agrain / 1e-4, dsigdlna[0, :])
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$a [cm]$')
plt.ylabel(r'$d\Sigma_d/d\ln a\; [\mathrm{g}\,\mathrm{cm}^{-3}]$')

finalize(results=(sigmadust))
