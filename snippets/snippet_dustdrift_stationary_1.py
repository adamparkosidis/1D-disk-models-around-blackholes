from snippet_header import DiskRadialModel, np, plt, MS, year, au, finalize

d = DiskRadialModel(rin=4 * au, rout=100 * au, nr=1000, alpha=2e-2)
d.make_disk_from_m_pl(1e-2 * MS, plsig=-1)
d.add_dust(agrain=1e0)

rbump     = 20. * au                     # Radial location of bump
abump     = 1.0                          # Amplitude of bump (relative)
hpbump    = np.interp(rbump, d.r, d.hp)  # Pressure scale height
wbump     = hpbump                       # Width (stand dev) of Gaussian bump
fact      = 1 + abump * np.exp(-0.5 * ((d.r - rbump) / wbump)**2)
d.sigma  *= fact
d.rhomid *= fact
mdotdust  = 1e-2 * 1e-10 * MS / year
d.dust[0].sigma = d.dust[0].get_drift_diffusion_solution(mdotdust)

plt.plot(d.r / au, d.sigma, label='Gas')
plt.plot(d.r / au, d.dust[0].sigma, label='Dust')
plt.xscale('log')
plt.yscale('log')
plt.xlim(d.r[0] / au, d.r[-1] / au)
plt.legend(loc='lower left')

finalize(results=(d.dust[0].sigma))
