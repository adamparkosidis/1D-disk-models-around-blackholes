from snippet_header import DiskRadialModel, plt, year, MS, au, finalize

d = DiskRadialModel(rout=400 * au)
d.make_disk_from_lbp_alpha(1e-2 * MS, 1 * au, 1e-2, 1e5 * year)
d.add_dust(agrain=1e-4)
lam = 0.13   # At 1.3 mm
d.dust[0].grain.compute_simple_opacity(lam, tabulatemean=False)
d.compute_onezone_intensity(lam)

plt.figure()
plt.plot(d.r / au, d.intensity[0, :])
plt.xlabel(r'$r [\mathrm{au}]$')
plt.ylabel(r'$I_\nu [\mathrm{CGS}]$')
plt.xscale('log')
plt.yscale('log')

print('Flux at one pc distance = {} Jy'.format(d.flux_at_oneparsec / 1e-23))
d.compute_tbright_from_intensity()
tbright_fullplanck = d.tbright.copy()
d.compute_tbright_from_intensity(linear=True)

plt.figure()
plt.plot(d.r / au, tbright_fullplanck[0, :], label='Full Planck')
plt.plot(d.r / au, d.tbright[0, :], label='Linear RJ')
plt.xlabel(r'$r [\mathrm{au}]$')
plt.ylabel(r'$T_{\mathrm{bright}} [K]$')
plt.xscale('log')
plt.yscale('log')
plt.ylim(bottom=1e-4)
plt.legend()

finalize(results=(tbright_fullplanck[0, :],d.tbright[0, :]))
